## Natural Language Processing

### 모델 및 데이터 다운로드

```bash
python download.py
```

---

## 🐳 Docker 서버 관리

### 서버 실행

```bash
# 이미지 빌드 및 서버 시작
sudo docker compose up --build -d

# 빌드 없이 서버 시작
sudo docker compose up -d
```

### 서버 중지 / 재시작

```bash
# 서버 중지
sudo docker compose down

# 서버 재시작
sudo docker compose restart

# 서버 중지 후 재시작
sudo docker compose down && sudo docker compose up -d
```

### 로그 확인

```bash
sudo docker compose logs -f
```

### 상태 확인

```bash
# 컨테이너 상태 확인
sudo docker compose ps

# 헬스 체크
curl http://localhost:8000/health
```

---

## 🚀 GPU 서버 설정 (배포 시)

`docker-compose.yml`에서 GPU 설정 주석 해제:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 📡 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 서버 정보 |
| GET | `/health` | 헬스 체크 (모델 상태, GPU 메모리) |
| POST | `/summarize/batch` | 배치 요약 및 번역 |
| POST | `/summarize/enrich` | 논문 한 편의 영문·한글 요약, 키워드, 임베딩 생성 |

### 요약 요청 예시

```bash
curl -X POST http://localhost:8000/summarize/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your paper text here..."]}'
```

### 논문 한 편의 AI 정보 생성

서버의 `models/`에 기존 요약·번역 모델과 함께 `scibert`, `all-mpnet-base-v2`를 둡니다. 요약 교정에 쓰는 `OPENAI_API_KEY`를 서버 환경에 설정합니다. 저장소 루트의 Compose로 실행할 때는 `nlp` 컨테이너 환경에 키를 전달합니다.

`abstract`는 필수 문자열입니다. 본문만 있으면 `abstract`에 빈 문자열을 넣고 `text`에 본문을 보냅니다. 초록만 있으면 `text`를 생략하거나 `null`을 보냅니다.

```bash
curl -X POST http://localhost:8000/summarize/enrich \
  -H "Content-Type: application/json" \
  -d '{"abstract": "Your paper abstract here...", "text": null}'
```

응답은 `summary.en`, `summary.ko`, `keywords`(문자열 배열), `embedding_vector`(정규화된 실수 배열)를 담습니다. 저장은 호출자가 담당합니다.
