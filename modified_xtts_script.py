
from TTS.api import TTS
import torch

# User-configurable target device ("cuda" or "cpu")
TARGET_DEVICE = "cuda"  # Or "cpu"

# Script parameters (placeholders for user input in Colab)
reference_wav_path = "/path/to/your/short_reference_clip.wav"  # Placeholder - User will define this
text_to_speak = "Hello, this is a test of your custom voice workflow with dynamic device selection."  # Placeholder - User will define this
output_wav_path = "xtts_generated_speech_dynamic_device.wav"
language_to_use = "en" # Adjust language if needed

# Determine the device to use
print(f"Target device specified: {TARGET_DEVICE}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if TARGET_DEVICE == "cuda" and cuda_available:
    device = "cuda"
elif TARGET_DEVICE == "cuda" and not cuda_available:
    device = "cpu"
    print("CUDA was targeted but is not available. Falling back to CPU.")
else:
    device = "cpu"

print(f"Using device: {device}")

# Init TTS model
try:
    print(f"Initializing TTS model on device: {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("TTS model initialized successfully.")

    # Generate speech using the reference audio
    print(f"Generating speech for text: \"{text_to_speak}\"")
    print(f"Using reference audio: {reference_wav_path}")
    tts.tts_to_file(
        text=text_to_speak,
        speaker_wav=reference_wav_path,
        language=language_to_use,
        file_path=output_wav_path
    )
    print(f"XTTS speech saved to {output_wav_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure that your reference_wav_path is correct and accessible.")
    if device == "cuda":
        print("If this is a CUDA-related error, you might be out of VRAM. Try using a smaller model or CPU.")

