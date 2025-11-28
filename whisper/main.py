import io
import time
from typing import Optional, Collection, List, Dict

import quart
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration

from whisper_model import WHISPER_MODEL_ID

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

app = quart.Quart(__name__)

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
model.config.forced_decoder_ids = None
tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL_ID)

if hasattr(torch, "compile"):
    print("Compiling model using PyTorch 2.0")
    torch.compile(processor)
    torch.compile(model)


def kill_extra_dynamic_range(pcm: torch.tensor):
    """
    Kill the extra dynamic range from audio greater than 16-bit bit depth.
    The model performs worse when the float values have more dynamic range
    than what would be possible when converting 16-bit int audio to float.
    """
    pcm_as_ints = (pcm * 32768).int()
    pcm_as_floats = pcm_as_ints.float() / 32768
    return pcm_as_floats


def detect_language(whisper_model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer, input_features,
                    possible_languages: Optional[Collection[str]] = None) -> List[Dict[str, float]]:
    # hacky, but all language tokens and only language tokens are 6 characters long
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    if possible_languages is not None:
        language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens) < len(possible_languages):
            raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    # 50258 is the token for transcribing
    logits = whisper_model(input_features,
                           decoder_input_ids=torch.tensor([[50258] for _ in range(input_features.shape[0])])).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')

    output_probs = logits.softmax(dim=-1).cpu()
    return [
        {
            lang: output_probs[input_idx, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for input_idx in range(logits.shape[0])
    ]


def inference(audio_bytes):
    try:
        pcm_channels, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception as e:
        raise RuntimeError(f"Failed to read audio data: {e}")
    pcm = torch.mean(pcm_channels, dim=0)
    pcm = kill_extra_dynamic_range(pcm)
    pcm = torchaudio.transforms.Resample(sample_rate, 16000)(pcm)

    # truncate silence via vad via torchaudio functional
    pcm_voice_sample = torchaudio.functional.vad(pcm, sample_rate=16000, trigger_level=0.1, trigger_time=0.1)
    pcm_voice_sample = pcm_voice_sample[:3 * 16000]

    # convert to numpy
    pcm = pcm.numpy()
    pcm_voice_sample = pcm_voice_sample.numpy()

    # convert to input features
    input_features = processor(pcm, 16000, return_tensors="pt").input_features
    input_features_sample = processor(pcm_voice_sample, 16000, return_tensors="pt").input_features

    # detect language from short sample
    language_probs = detect_language(model, tokenizer, input_features_sample)
    argmax_language = max(language_probs[0], key=language_probs[0].get)[2:-2]

    # set forced decoder ids to language token
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=argmax_language, task="transcribe")

    # run inference
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]


@app.route("/api/v1/transcribe", methods=["POST"])
async def transcribe():
    """
    The expected input Content-Type is audio/mpeg, audio/wav, audio/ogg, or audio/flac.
    :return: A JSON object with the transcription.
    """
    if quart.request.method == "POST":
        audio_bytes = await quart.request.data
        if not isinstance(audio_bytes, bytes):
            return quart.jsonify({"error": "The audio data must be bytes."}), 400
        start_time = time.time()
        result = inference(audio_bytes)
        end_time = time.time()
        print("Inference took", end_time - start_time, "seconds.")
        return quart.jsonify({"transcription": result}), 200
    return None


@app.route("/health", methods=["GET"])
async def health_check():
    return quart.jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    print("Starting Whisper transcription server...")
    print("Using device:", device)
    app.run(host="0.0.0.0", port=5001)
