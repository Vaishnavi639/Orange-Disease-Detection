FROM python:3.9-slim AS builder
WORKDIR /install
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache

FROM python:3.9-slim
LABEL maintainer="Orange Disease Detection" 
ENV PYTHONUNBUFFERED=1
RUN adduser --disabled-password --gecos '' appuser 
USER appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages 
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --chown=appuser:appuser app.py . 
COPY --chown=appuser:appuser config/ ./config 
COPY --chown=appuser:appuser utils/ ./utils 
COPY --chown=appuser:appuser models/ ./models 
COPY --chown=appuser:appuser saved_models/ ./saved_models 
COPY --chown=appuser:appuser frontend/ ./frontend

# Create uploads directory if needed
USER root
RUN mkdir -p /app/frontend/uploads && chown -R appuser:appuser /app/frontend/uploads
USER appuser

EXPOSE 5000
CMD ["python", "-u", "app.py"]