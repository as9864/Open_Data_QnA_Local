# ---------- Stage 1: build ----------
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 POETRY_VERSION=1.8.3 POETRY_VIRTUALENVS_CREATE=false PYTHONUNBUFFERED=1
WORKDIR /app

# 의존성 메타만 복사 (캐시 극대화)
COPY pyproject.toml poetry.lock* ./

# poetry + export 플러그인
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" poetry-plugin-export

# 현재 환경에 맞게 lock 재계산(이전 triton 고정 등 깨끗이 정리)
RUN poetry lock -n

# requirements 생성
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# ---------- Stage 2: runtime (CPU 전용 설치) ----------
FROM python:3.12-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 APP_HOME=/app TRANSFORMERS_CACHE=/cache
WORKDIR $APP_HOME

# (기존 내용 유지)
COPY --from=build /app/requirements.txt /requirements.txt

# 🔧 CPU에 불필요/문제 되는 항목 제거(스트림릿 고정/ GPU 패키지/트리톤)
RUN sed -i '/^streamlit==/d' /requirements.txt && \
    sed -i '/^nvidia-/d' /requirements.txt && \
    sed -i '/^triton==/d' /requirements.txt && \
    sed -i '/cublas\|cudnn\|cuda\|cufft\|cufile\|curand\|cusolver\|cusparse\|cusparselt\|nccl\|nvjitlink\|nvtx/d' /requirements.txt

# ✅ CPU 전용 인덱스로 설치
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /requirements.txt
# 앱 소스 복사 (모델/캐시/데이터는 .dockerignore로 제외)
COPY . .

# 혹시 생겼을 캐시/바이트코드 제거
RUN rm -rf /root/.cache ~/.cache || true && \
    find / -type d -name "__pycache__" -prune -exec rm -rf {} + || true

# 모델 캐시는 이미지 밖으로 (컨테이너 실행 시 마운트)
VOLUME ["/cache"]

# Cloud Run: $PORT 사용 (JSON 배열 CMD로 신호 처리 안전)
#ENV PORT=8080
#CMD ["gunicorn","--bind",":${PORT:-8080}","--workers","1","--threads","8","--timeout","0","backend-apis.main:app"]

CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 0 backend-apis.main:app

