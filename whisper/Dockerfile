FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

# install packages for health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg via conda-forge for proper torchcodec compatibility
# Pin to FFmpeg 7.x which is supported by torchcodec
RUN conda install -y 'ffmpeg>=7,<8' -c conda-forge

# Set LD_LIBRARY_PATH so torchcodec can find FFmpeg libraries at runtime
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

COPY . /app
WORKDIR /app

# install regular dependencies
RUN pip3 install -r requirements.txt --no-cache-dir

# Install torchaudio and torchcodec with matching versions for PyTorch 2.9.1
RUN pip3 install torchaudio==2.9.1 torchcodec==0.8.1 --extra-index-url https://download.pytorch.org/whl/cu130

#RUN pip3 install --force-reinstall markupsafe==2.0.1

EXPOSE 5001

CMD ["python3", "main.py"]