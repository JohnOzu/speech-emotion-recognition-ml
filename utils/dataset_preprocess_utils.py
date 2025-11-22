import librosa
import soundfile as sf
import os
import re
import numpy as np

RAW_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed/audio_16k"
TARGET_SR = 16000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)

def process_file(filepath, rows):
    try:
        audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)

        # Trim leading and trailing silence (optional but good)
        audio, _ = librosa.effects.trim(audio)

        # Pad or Trim to fixed length
        if len(audio) < TARGET_LENGTH:
            pad_width = TARGET_LENGTH - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant")

        else:
            audio = audio[:TARGET_LENGTH]

        filename = os.path.basename(filepath)
        base = os.path.splitext(filename)[0]

        base_parts = re.split("_", base)
        dataset_name = base_parts[0]
        actor_id = base_parts[1]
        emotion = base_parts[2]
        index = base_parts[3]

        out_name = f"{base}_16k.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        sf.write(out_path, audio, TARGET_SR)

        rows.append([out_path, emotion, dataset_name, actor_id, index, TARGET_SR])

    except Exception as e:
        print("Error:", e, filepath)