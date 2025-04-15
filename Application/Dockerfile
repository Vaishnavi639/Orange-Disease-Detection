FROM python:3.9-slim

LABEL maintainer="Orange Disease Detection"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Copy app files (excluding unnecessary images or files)
COPY app.py .
COPY config ./config
COPY utils ./utils
COPY models ./models
COPY saved_models ./saved_models
COPY frontend ./frontend

# Remove large unnecessary image if exists
RUN rm -f frontend/uploaded_image.jpg

# Create non-root user
RUN adduser --disabled-password --no-create-home appuser
USER appuser

EXPOSE 5000

CMD ["python", "app.py"]

