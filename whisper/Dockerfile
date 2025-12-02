FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# install packages for health check and audio processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

# install regular dependencies
RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install torchaudio --extra-index-url https://download.pytorch.org/whl/cu129

EXPOSE 5001

CMD ["python3", "main.py"]