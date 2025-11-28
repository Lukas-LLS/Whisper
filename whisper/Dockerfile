FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

# install packages for health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg via conda-forge for proper torchcodec compatibility
RUN conda install -y ffmpeg -c conda-forge

COPY . /app
WORKDIR /app

# install regular dependencies
RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cu130

#RUN pip3 install --force-reinstall markupsafe==2.0.1

EXPOSE 5001

CMD ["python3", "main.py"]