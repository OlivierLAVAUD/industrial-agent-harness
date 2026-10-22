FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY pyproject.toml README.md policy.yaml ./
COPY graph ./graph
COPY agent ./agent
COPY harness ./harness
COPY ui ./ui
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install ".[dev]"

COPY tests ./tests

RUN mkdir -p /app/var /root/.streamlit \
    && printf '[server]\nheadless = true\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml \
    && printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
