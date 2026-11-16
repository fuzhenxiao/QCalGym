FROM python:3.11-slim AS wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /requirements.txt

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
COPY --from=wheels /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
COPY backend backend
COPY benchmark benchmark
COPY tasks tasks
COPY scenes scenes
COPY client.py .
EXPOSE 8000
ENTRYPOINT ["python", "-m", "benchmark"]
CMD ["list"]
