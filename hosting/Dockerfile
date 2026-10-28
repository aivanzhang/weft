FROM python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY hosting/requirements.lock /app/hosting/requirements.lock
RUN pip install --no-cache-dir -r hosting/requirements.lock && useradd --uid 10001 --create-home weft
COPY src /app/src
COPY pyproject.toml requirements.lock /app/
COPY hosting/app.py hosting/job.py /app/hosting/
USER 10001
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "hosting.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log", "--limit-concurrency", "4", "--timeout-keep-alive", "5"]
