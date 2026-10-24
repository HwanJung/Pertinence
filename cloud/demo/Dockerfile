# syntax=docker/dockerfile:1.7
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /opt/pertinence
COPY cloud/demo/requirements.lock /tmp/demo-requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/demo-requirements.lock \
    && python -m pip check
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -c 'from pertinence.demo.app import create_app; create_app()' \
    && rm /tmp/demo-requirements.lock
USER 65532:65532
EXPOSE 8080
CMD ["python", "-m", "pertinence.demo"]
