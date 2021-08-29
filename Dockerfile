FROM python:3.9.5-slim-buster AS py3
FROM openjdk:8-slim-buster

COPY --from=py3 / /

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app/

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge