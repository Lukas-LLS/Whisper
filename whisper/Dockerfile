FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

# install packages for health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg 6.x via conda-forge for torchcodec compatibility
# FFmpeg 6.x is more stable with torchcodec than FFmpeg 7.x
RUN conda install -y 'ffmpeg>=6,<7' -c conda-forge

# Set LD_LIBRARY_PATH so torchcodec can find FFmpeg libraries at runtime
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

# Verify FFmpeg libraries are available
RUN ls -la /opt/conda/lib/libav* || true

COPY . /app
WORKDIR /app

# install regular dependencies
RUN pip3 install -r requirements.txt --no-cache-dir

# Install torchaudio and torchcodec with matching versions for PyTorch 2.9.1
# Use --no-build-isolation to ensure torchcodec can find FFmpeg during installation
RUN pip3 install torchaudio==2.9.1 torchcodec==0.8.1 --extra-index-url https://download.pytorch.org/whl/cu130

# Verify torchcodec can load at build time
RUN python3 -c "import torchcodec; print('torchcodec loaded successfully')" || echo "WARNING: torchcodec failed to load"

#RUN pip3 install --force-reinstall markupsafe==2.0.1

EXPOSE 5001

CMD ["python3", "main.py"]