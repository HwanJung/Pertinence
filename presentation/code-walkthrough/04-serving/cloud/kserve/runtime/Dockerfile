# syntax=docker/dockerfile:1.7
# Override with a verified linux/amd64 Python 3.11 slim image. The sentinel is rejected by
# build-runtime.sh and must never be pushed or deployed.
ARG PYTHON_BASE_IMAGE=python:3.11-slim@sha256:0000000000000000000000000000000000000000000000000000000000000000
FROM ${PYTHON_BASE_IMAGE}
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ORT_INTRA_OP_NUM_THREADS=1 \
    ORT_INTER_OP_NUM_THREADS=1 \
    PERTINENCE_WORKERS=1

WORKDIR /opt/pertinence
COPY cloud/kserve/runtime/requirements.lock /tmp/serving-requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/serving-requirements.lock
RUN test "$TARGETARCH" = "amd64" \
    && python -c 'import sys; assert sys.version_info[:2] == (3, 11)'

COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip install --no-cache-dir --no-deps . \
    && rm -rf /root/.cache /tmp/serving-requirements.lock \
    && chown -R 65532:65532 /opt/pertinence

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["pertinence-serving"]
CMD ["--help"]
