import unittest
from contextlib import ExitStack
from unittest.mock import Mock, call, patch

from fastapi.testclient import TestClient

import server
from app.pipeline import pipeline


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)

        self.summarizer = Mock(device="cpu")
        self.translator = Mock(device="cpu")
        self.keyword_model = Mock()
        self.embedding_model = Mock()
        self.refiner = Mock()

        self.summarizer.summarize.return_value = "Raw English summary."
        self.translator.translate.return_value = "한국어 초벌 요약."
        self.keyword_model.extract_keywords.return_value = [("graph learning", 0.9), ("graphs", 0.8)]
        self.embedding_model.encode.return_value = [Mock(tolist=Mock(return_value=[0.6, 0.8]))]
        self.refiner.refine_both.return_value = (" Refined English summary. ", " 다듬은 한국어 요약. ")

        get_summarizer = Mock(return_value=self.summarizer)
        get_translator = Mock(return_value=self.translator)
        stack.enter_context(patch.multiple(server, get_summarizer=get_summarizer, get_translator=get_translator))
        self.get_enrichment_models = Mock(return_value=(self.keyword_model, self.embedding_model, self.refiner))
        stack.enter_context(patch.multiple(
            pipeline,
            get_summarizer=get_summarizer,
            get_translator=get_translator,
            _get_enrichment_models=self.get_enrichment_models,
        ))
        self.client = stack.enter_context(TestClient(server.app))

    def test_enrich_uses_abstract_and_body_and_returns_paper_fields(self):
        abstract = "A" * 3950
        body = "Paper body. " * 10
        self.summarizer.summarize.side_effect = [
            "Abstract summary.", "Body summary.", "@artifact Raw English summary.",
        ]

        response = self.client.post("/summarize/enrich", json={"abstract": abstract, "text": body})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "summary": {"en": "Refined English summary.", "ko": "다듬은 한국어 요약."},
            "keywords": ["graph learning", "graphs"],
            "embedding_vector": [0.6, 0.8],
        })
        self.assertEqual(self.summarizer.summarize.call_args_list, [
            call(abstract), call(body.strip()), call("Abstract summary. Body summary."),
        ])
        self.translator.translate.assert_called_once_with("Raw English summary.")
        self.keyword_model.extract_keywords.assert_called_once_with(
            "Raw English summary.", keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10,
        )
        self.refiner.refine_both.assert_called_once_with(
            summary_en_raw="Raw English summary.",
            summary_ko_raw="한국어 초벌 요약.",
            keywords_en=["graph learning", "graphs"],
        )
        self.embedding_model.encode.assert_called_once_with(
            ["graph learning\ngraphs"], normalize_embeddings=True,
        )

    def test_text_can_be_omitted_or_null(self):
        for body in ({}, {"text": None}):
            with self.subTest(body=body):
                self.summarizer.summarize.reset_mock()
                response = self.client.post("/summarize/enrich", json={"abstract": "Abstract.", **body})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.summarizer.summarize.call_args_list, [
                    call("Abstract."), call("Raw English summary."),
                ])

    def test_text_only_paper_can_send_empty_abstract(self):
        response = self.client.post("/summarize/enrich", json={"abstract": "", "text": "Body."})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.summarizer.summarize.call_args_list, [
            call("Body."), call("Raw English summary."),
        ])

    def test_abstract_is_required_and_either_field_must_contain_text(self):
        invalid_payloads = (
            {"text": "Body."},
            {"abstract": None, "text": "Body."},
            {"abstract": ""},
            {"abstract": " \n ", "text": None},
            {"abstract": " ", "text": " \n "},
        )
        for request in invalid_payloads:
            with self.subTest(request=request):
                self.assertEqual(self.client.post("/summarize/enrich", json=request).status_code, 422)
        self.summarizer.summarize.assert_not_called()
        self.get_enrichment_models.assert_not_called()

    def test_missing_refined_language_fails_before_embedding(self):
        for summaries in ((" ", "한국어 요약."), ("English summary.", "\n")):
            with self.subTest(summaries=summaries):
                self.refiner.refine_both.return_value = summaries
                with self.assertLogs("server", "ERROR"):
                    response = self.client.post("/summarize/enrich", json={"abstract": "Abstract."})
                self.assertEqual(response.status_code, 500)
        self.embedding_model.encode.assert_not_called()

    def test_unusable_embedding_fails(self):
        for vector in ([], [float("nan")]):
            with self.subTest(vector=vector):
                self.embedding_model.encode.return_value[0].tolist.return_value = vector
                with self.assertLogs("server", "ERROR"):
                    response = self.client.post("/summarize/enrich", json={"abstract": "Abstract."})
                self.assertEqual(response.status_code, 500)

    def test_batch_contract_and_shared_inference_lock(self):
        def assert_lock_and_return(summaries):
            self.assertTrue(server.inference_lock.locked())
            return summaries

        self.summarizer.summarize_batch.side_effect = lambda _: assert_lock_and_return(["One.", "Two."])
        self.translator.translate_batch.return_value = ["하나.", "둘."]
        response = self.client.post("/summarize/batch", json={"texts": ["First.", "Second."]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": [
            {"summary_en": "One.", "summary_ko": "하나."},
            {"summary_en": "Two.", "summary_ko": "둘."},
        ]})
        self.get_enrichment_models.assert_not_called()

        def refine(**_):
            self.assertFalse(server.inference_lock.locked())
            return "Refined English summary.", "다듬은 한국어 요약."

        self.summarizer.summarize.side_effect = lambda _: assert_lock_and_return("English summary.")
        self.refiner.refine_both.side_effect = refine
        embedding = self.embedding_model.encode.return_value
        self.embedding_model.encode.side_effect = lambda texts, normalize_embeddings: assert_lock_and_return(embedding)
        response = self.client.post("/summarize/enrich", json={"abstract": "Abstract."})
        self.assertEqual(response.status_code, 200)
