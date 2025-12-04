import librosa
import numpy as np

def extract_mfcc_features(y, sr, n_mfcc=40, hop_length=512, n_fft=1024):
    
    # Extract MFCC features - captures timbral texture of speech.
    # MFCCs are excellent for capturing phonetic content and vocal tract shape.
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Delta MFCCs (velocity)
    mfcc_delta = librosa.feature.delta(mfccs)
    
    # Delta-delta MFCCs (acceleration)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    all_mfccs = np.vstack([mfccs, mfcc_delta, mfcc_delta2])

    return all_mfccs

def extract_mel_spectrogram(y, sr, n_mels=128, hop_length=512, n_fft=2048, fmin=0, fmax=None):
    
    # Mel-spectrogram is a time-frequency representation that mimics 
    # human auditory perception. Better than MFCC for capturing 
    # emotional nuances in voice quality.
    
    # Extract mel-spectrogram (power spectrogram)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Power spectrum (magnitude squared)
    )
    
    # Convert to log scale (dB)
    # Human perception of loudness is logarithmic
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec  # Shape: (n_mels, time_frames)

def extract_prosodic_features(y, sr, target_frames=None):
    # Extract pitch and energy - helps distinguish fear/happy/sad

    # Pitch (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        frame_length=2048
    )
    f0 = np.nan_to_num(f0)

    # Energy (RMS) 
    rms = librosa.feature.rms(y=y, hop_length=512)[0]  # Shape: (time_frames,)
    
    # Ensure all have same length
    min_len = min(len(f0), len(rms))
    f0 = f0[:min_len]
    rms = rms[:min_len]

    prosodic_features = np.vstack([
        f0,
        rms,
    ])
    
    return prosodic_features

def extract_spectral_features(y, sr, hop_length=512):

    # Spectral features - frequency domain characteristics.

    # Spectral Centroid - "brightness"
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Spectral Rolloff - high-frequency content
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Spectral Contrast - harmonic structure
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length, n_bands=6)
    
    # Spectral Bandwidth - frequency spread
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Spectral Flatness - noise-like vs tonal
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    
    # Align lengths
    min_len = min(len(centroid), len(rolloff), contrast.shape[1], 
                  len(bandwidth), len(flatness))
    
    return np.vstack([
        centroid[:min_len],
        rolloff[:min_len],
        contrast[:, :min_len],
        bandwidth[:min_len],
        flatness[:min_len]
    ])  # Shape: (12, time_frames)

def augment_audio(file_path, sr):
    # Apply multiple augmentations to one audio sample

    audio, _ = librosa.load(file_path, sr=sr)

    augmented_samples = []

    # Original (no augmentation)
    augmented_samples.append((audio, "original"))

    # Time stretching
    audio_stretch = librosa.effects.time_stretch(audio, rate=0.9)
    augmented_samples.append((audio_stretch, "time_stretch_0.9"))

    # Pitch shifting
    audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    augmented_samples.append((audio_pitch, "pitch_shift_2"))

    # Add noise
    noise = np.random.randn(len(audio)) * 0.005
    audio_noise = audio + noise
    augmented_samples.append((audio_noise, "noise_0.005"))

    # Audio Scaling
    augmented_samples.append((audio * 1.2, "volume_1.2"))
    
    return augmented_samples

def compute_feature_statistics(features):
    
    # Args:
        # features: 2D array of shape (n_features, time_frames)
    
    # Returns:
        # 1D array of statistical features
    
    stats = []
    a
    # Compute statistics across time (axis=1)
    stats.append(np.mean(features, axis=1))      # Mean
    stats.append(np.std(features, axis=1))       # Standard deviation
    stats.append(np.max(features, axis=1))       # Maximum
    stats.append(np.min(features, axis=1))       # Minimum
    stats.append(np.median(features, axis=1))    # Median
    
    # Optional: Add percentiles
    stats.append(np.percentile(features, 25, axis=1))  # 25th percentile
    stats.append(np.percentile(features, 75, axis=1))  # 75th percentile
    
    # Concatenate all statistics
    feature_vector = np.concatenate(stats)
    
    return feature_vector  # Shape: (n_features * 7,)

def pad_features(features, max_len=150):
    
    #Pad or truncate to fixed length.

    if features.ndim != 2:
        raise ValueError(f"Expected 2D array, got {features.ndim}D")

    current_len = features.shape[1]
        
    if current_len < max_len:
        # Pad with zeros
        pad_width = max_len - current_len
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    elif current_len > max_len:
        # Truncate
        features = features[:, :max_len]
    
    return features

def normalize_features(features):

    # Normalize MFCC features.

    mean = np.mean(features)
    std = np.std(features)

    if std < 1e-8:
        # Return zero-centered features without dividing by zero
        return features - mean
    
    normalized = (features - mean) / (std + 1e-8)

    # CLIP extreme values to prevent issues
    normalized = np.clip(normalized, -5, 5)
    
    return normalized