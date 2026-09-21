# syntax=docker/dockerfile:1.7
# BusyBox is required because KServe v0.20.0 modelcar invokes sh, ln, and sleep.
ARG MODELCAR_BASE_IMAGE=busybox:1.37.0@sha256:0000000000000000000000000000000000000000000000000000000000000000
FROM ${MODELCAR_BASE_IMAGE}

RUN mkdir -p /models && chmod 0555 /models
COPY --chown=65532:65532 bundle/ /models/
USER 65532:65532

