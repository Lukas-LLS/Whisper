FROM pytorch/pytorch

COPY . /app
WORKDIR /app

# install regular dependencies
RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

#RUN pip3 install --force-reinstall markupsafe==2.0.1

EXPOSE 5001

# run download_model.py in container to package whisper model
RUN python3 download_model.py

CMD ["python3", "main.py"]