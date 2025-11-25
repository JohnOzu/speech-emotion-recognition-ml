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
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    
    # Spectral Rolloff 
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
    
    # Ensure all have same length
    min_len = min(len(f0), len(rms), len(zcr), len(spectral_centroid), len(spectral_rolloff))

    f0 = f0[:min_len]
    rms = rms[:min_len]
    zcr = zcr[:min_len]
    spectral_centroid = spectral_centroid[:min_len]
    spectral_rolloff = spectral_rolloff[:min_len]

    prosodic_features = np.vstack([
        f0,
        rms,
        zcr,
        spectral_centroid,
        spectral_rolloff
    ])

    if target_frames is not None and prosodic_features.shape[1] != target_frames:
        from scipy import interpolate
        x_old = np.linspace(0, 1, prosodic_features.shape[1])
        x_new = np.linspace(0, 1, target_frames)
        
        prosodic_resampled = np.zeros((prosodic_features.shape[0], target_frames))
        for i in range(prosodic_features.shape[0]):
            f = interpolate.interp1d(x_old, prosodic_features[i, :], kind='linear')
            prosodic_resampled[i, :] = f(x_new)
        
        prosodic_features = prosodic_resampled
    
    return prosodic_features

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
    augmented_samples.append((audio_pitch, "noise_0.005"))
    
    return augmented_samples

def pad_features(features, max_len=150):
    
    #Pad or truncate to fixed length.
        
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]
    
    return features

def normalize_features(features):

    # Normalize MFCC features.

    mean = np.mean(features)
    std = np.std(features)
    normalized = (features - mean) / (std + 1e-8)

    # CLIP extreme values to prevent issues
    normalized = np.clip(normalized, -5, 5)
    
    return normalized