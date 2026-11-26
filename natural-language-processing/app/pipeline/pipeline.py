from functools import cache
from threading import Lock

from app.nlp.summarizer import get_summarizer
from app.nlp.translator import get_translator
from app.pipeline.text_utils import chunk_text, clean_summary_en


inference_lock = Lock()


@cache
def _get_enrichment_models():
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    from app.nlp.refiner import LocalKoreanRefiner

    refiner = LocalKoreanRefiner()
    keyword_model = KeyBERT("models/scibert")
    embedding_model = SentenceTransformer("models/all-mpnet-base-v2")
    return keyword_model, embedding_model, refiner


def enrich_paper(abstract: str, text: str | None = None) -> dict:
    raw_text = "\n".join((abstract, text or "")).strip()
    with inference_lock:
        summarizer = get_summarizer()
        chunk_summaries = [
            summarizer.summarize(chunk).strip()
            for chunk in chunk_text(raw_text, max_chars=4000)
        ]
        summary_input = " ".join(summary for summary in chunk_summaries if summary)
        summary_en = clean_summary_en(summarizer.summarize(summary_input))
        if not summary_en:
            raise ValueError("English summarization returned no text")

        summary_ko = get_translator().translate(summary_en).strip()
        if not summary_ko:
            raise ValueError("Korean translation returned no text")

        keyword_model, embedding_model, refiner = _get_enrichment_models()
        keywords = [
            word
            for word, _ in keyword_model.extract_keywords(
                summary_en,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=10,
            )
        ]

    refined_en, refined_ko = refiner.refine_both(
        summary_en_raw=summary_en,
        summary_ko_raw=summary_ko,
        keywords_en=keywords,
    )
    refined_en, refined_ko = refined_en.strip(), refined_ko.strip()
    if not refined_en or not refined_ko:
        raise ValueError("Refinement must return both English and Korean summaries")

    with inference_lock:
        embedding = embedding_model.encode(
            ["\n".join(keywords).strip()],
            normalize_embeddings=True,
        )[0]

    return {
        "summary": {"en": refined_en, "ko": refined_ko},
        "keywords": keywords,
        "embedding_vector": embedding.tolist(),
    }
