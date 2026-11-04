FROM python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    WEFT_POSE_MODEL=/app/models/pose_landmarker_lite.task
WORKDIR /app
# MediaPipe pulls the non-headless OpenCV, which needs GL/GLib at import, and sounddevice
# needs PortAudio even though no audio is used. MuJoCo runs physics only; no GPU driver.
# OpenCV stays on 4.14 so the hosted decoder matches the locally verified lab exactly;
# 5.0 produced the same admissions in a 7-clip sweep but is an unverified decode path.
RUN apt-get update && apt-get install --no-install-recommends --yes \
        libgl1 libegl1 libgles2 libosmesa6 libglib2.0-0 libportaudio2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY hosting/requirements.lock /app/hosting/requirements.lock
RUN pip install --no-cache-dir -r hosting/requirements.lock && useradd --uid 10001 --create-home weft
# The pinned pose model is fetched at build time and hash-checked, so the image never
# depends on an untracked local file and the weights stay verifiable.
ADD --checksum=sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task \
    /app/models/pose_landmarker_lite.task
# ADD leaves the weights root-owned and unreadable to the unprivileged runtime user.
RUN chmod -R a+rX /app/models
COPY src /app/src
COPY pyproject.toml requirements.lock /app/
COPY hosting/app.py hosting/job.py hosting/exercise_job.py /app/hosting/
USER 10001
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "hosting.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log", "--limit-concurrency", "4", "--timeout-keep-alive", "5"]
