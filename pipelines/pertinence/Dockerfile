FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pertinence
COPY requirements.lock pyproject.toml README.md ./
COPY pipelines/pertinence/requirements-components.lock ./requirements-components.lock
COPY src ./src

RUN python -m pip install --requirement requirements.lock \
    && python -m pip install --requirement requirements-components.lock \
    && python -m pip install . --no-deps

ENTRYPOINT ["python", "-m", "pertinence.pipeline_cli"]
