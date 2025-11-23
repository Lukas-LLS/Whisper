"""
Download the whisper model from the Hugging Face model hub.
This is called by the docker build script to package the model with the docker image.
"""
from huggingface_hub import hf_hub_download

from whisper_model import WHISPER_MODEL_ID

if __name__ == '__main__':
    print("Downloading model from Hugging Face model hub...")
    hf_hub_download(WHISPER_MODEL_ID, "pytorch_model.bin")
    print("Done.")
