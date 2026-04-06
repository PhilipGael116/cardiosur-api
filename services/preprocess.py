import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from math import gcd

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE    = 500   # model expects 500 Hz
MODEL_LOW_FREQ        = 20    # model bandpass lower bound (Hz) — matches training
MODEL_HIGH_FREQ       = 500   # model bandpass upper bound (Hz) — matches training
AUSCULTATION_LOW_FREQ = 20    # cardiac auscultation lower bound (Hz)
AUSCULTATION_HIGH_FREQ= 150   # S1/S2 "lub-dub" upper bound (Hz)
CHUNK_SECONDS         = 2     # window size the model was trained on
CHUNK_SAMPLES         = TARGET_SAMPLE_RATE * CHUNK_SECONDS  # 1000 samples/chunk
SPEC_SIZE             = 128   # model expects 128×128 spectrogram
NOISE_GATE_PERCENTILE = 15    # quietest % of frames used to estimate noise floor


# ── Step 1: Resampling ────────────────────────────────────────────────────────

def resample_audio(audio: np.ndarray, original_sr: int) -> np.ndarray:
    """
    Resample audio from original sample rate to 500 Hz using polyphase filtering.
    """
    if original_sr == TARGET_SAMPLE_RATE:
        return audio

    common = gcd(original_sr, TARGET_SAMPLE_RATE)
    up   = TARGET_SAMPLE_RATE // common
    down = original_sr // common

    return resample_poly(audio, up, down).astype(np.float32)


# ── Step 2: Noise Suppression ─────────────────────────────────────────────────

def spectral_noise_gate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Spectral subtraction noise suppression.

    Estimates the noise floor from the quietest frames of the recording
    (background room noise) and subtracts it from all frames, leaving only
    the cardiac signal components.
    """
    n_fft      = 256
    hop_length = 64

    stft      = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase     = np.angle(stft)

    # Identify quiet / noise-only frames
    frame_energy  = np.mean(magnitude, axis=0)
    noise_thresh  = np.percentile(frame_energy, NOISE_GATE_PERCENTILE)
    noise_frames  = magnitude[:, frame_energy <= noise_thresh]

    noise_profile = (
        np.mean(noise_frames, axis=1, keepdims=True)
        if noise_frames.shape[1] > 0
        else np.min(magnitude, axis=1, keepdims=True)
    )

    # Over-subtract (α = 2) and floor at 10 % of original to avoid musical noise
    alpha            = 2.0
    magnitude_clean  = np.maximum(
        magnitude - alpha * noise_profile,
        0.1 * magnitude
    )

    stft_clean   = magnitude_clean * np.exp(1j * phase)
    audio_clean  = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))

    return audio_clean.astype(np.float32)


# ── Step 3: Bandpass Filtering ────────────────────────────────────────────────

def bandpass_filter(
    audio:       np.ndarray,
    sample_rate: int,
    low_freq:    int = MODEL_LOW_FREQ,
    high_freq:   int = MODEL_HIGH_FREQ,
) -> np.ndarray:
    """
    4th-order Butterworth band-pass filter.

    Default range (20–500 Hz) matches model training data.
    Pass AUSCULTATION_LOW/HIGH for S1/S2-focused analysis.
    """
    nyquist = sample_rate / 2
    low     = low_freq  / nyquist
    high    = min(high_freq / nyquist, 0.99)   # never exceed Nyquist

    b, a     = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, audio)

    return filtered.astype(np.float32)


# ── Step 4: Standardisation ───────────────────────────────────────────────────

def standardize(audio: np.ndarray) -> np.ndarray:
    """
    Standardise signal to zero mean and unit standard deviation.
    """
    mean = np.mean(audio)
    std  = np.std(audio)

    return ((audio - mean) / std) if std > 0 else (audio - mean)


# ── Step 5: Shannon Energy Envelope ──────────────────────────────────────────

def extract_shannon_envelope(audio: np.ndarray) -> np.ndarray:
    """
    Compute the Shannon Energy envelope.

    E[n] = -x[n]^2 · log(x[n]^2)

    Shannon Energy suppresses high-amplitude noise spikes while amplifying
    medium-energy cardiac events (S1/S2), making it more reliable than
    raw amplitude or Hilbert-based envelopes for phonocardiograms.
    """
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return np.zeros_like(audio, dtype=np.float32)

    x    = audio / max_val
    x_sq = np.clip(x ** 2, 1e-10, None)          # avoid log(0)
    shannon_energy = -x_sq * np.log(x_sq)

    # Smooth with a 50 ms moving average
    window_size = max(1, int(0.05 * TARGET_SAMPLE_RATE))
    window      = np.ones(window_size) / window_size
    envelope    = np.convolve(shannon_energy, window, mode="same")

    return envelope.astype(np.float32)


# ── Step 5b: BPM Estimation ───────────────────────────────────────────────────

def estimate_bpm_from_envelope(audio: np.ndarray, sample_rate: int) -> int:
    """
    Estimate BPM by detecting S1 peaks in the Shannon Energy envelope.

    Uses the auscultation bandpass (20–150 Hz) to isolate the fundamental
    "lub-dub" cardiac sounds before envelope extraction, giving a far more
    accurate heart rate than raw amplitude thresholding.
    """
    # Isolate S1/S2 frequency band first
    filtered = bandpass_filter(
        audio, sample_rate,
        AUSCULTATION_LOW_FREQ, AUSCULTATION_HIGH_FREQ
    )

    envelope = extract_shannon_envelope(filtered)

    # Peak detection: min 0.3 s gap (≈ 200 BPM max), height ≥ 30 % of max
    min_distance = int(0.3 * sample_rate)
    threshold    = 0.30 * np.max(envelope)

    peaks, _ = find_peaks(envelope, distance=min_distance, height=threshold)

    if len(peaks) < 2:
        return _estimate_bpm_simple(audio, sample_rate)

    # Use median inter-peak interval for robustness against outliers
    intervals   = np.diff(peaks) / sample_rate   # seconds
    avg_interval = np.median(intervals)
    bpm          = int(60 / avg_interval)

    return max(40, min(bpm, 200))


def _estimate_bpm_simple(audio: np.ndarray, sample_rate: int) -> int:
    """
    Fallback amplitude-threshold BPM estimation (used when envelope gives
    fewer than 2 detectable peaks).
    """
    threshold = np.mean(np.abs(audio)) + np.std(np.abs(audio))
    above     = np.abs(audio) > threshold

    beats, in_beat = 0, False
    min_gap, last_beat = sample_rate * 0.3, -(sample_rate * 0.3)

    for i, val in enumerate(above):
        if val and not in_beat and (i - last_beat) > min_gap:
            beats    += 1
            in_beat   = True
            last_beat = i
        elif not val:
            in_beat = False

    duration_minutes = len(audio) / sample_rate / 60
    bpm = int(beats / duration_minutes) if duration_minutes > 0 else 0

    return max(40, min(bpm, 200))


# ── Step 6: Chunking ──────────────────────────────────────────────────────────

def split_into_chunks(audio: np.ndarray) -> list:
    """
    Split audio into non-overlapping 2-second chunks (1 000 samples each).
    Incomplete final chunks are discarded.
    """
    return [
        audio[start : start + CHUNK_SAMPLES]
        for start in range(0, len(audio), CHUNK_SAMPLES)
        if len(audio[start : start + CHUNK_SAMPLES]) == CHUNK_SAMPLES
    ]


# ── Step 7: Mel Spectrogram ───────────────────────────────────────────────────

def chunk_to_mel_spectrogram(chunk: np.ndarray) -> np.ndarray:
    """
    Convert a 2-second audio chunk to a 128×128 mel spectrogram.
    Parameters deliberately match the model's training configuration.

    Pipeline:
        1. Mel spectrogram  (n_mels=128, fmax=500 Hz)
        2. Power → dB conversion (log scale)
        3. Resize to 128×128
        4. Normalise to [0, 1]
        5. Reshape to (128, 128, 1)  ← single-channel for the model
    """
    mel    = librosa.feature.melspectrogram(
        y=chunk, sr=TARGET_SAMPLE_RATE, n_mels=128, fmax=MODEL_HIGH_FREQ
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    from PIL import Image
    mel_resized = np.array(Image.fromarray(mel_db).resize((SPEC_SIZE, SPEC_SIZE)))

    mel_min, mel_max = mel_resized.min(), mel_resized.max()
    if mel_max - mel_min == 0:
        mel_norm = np.zeros_like(mel_resized, dtype=np.float32)
    else:
        mel_norm = (mel_resized - mel_min) / (mel_max - mel_min)

    return mel_norm.reshape(SPEC_SIZE, SPEC_SIZE, 1).astype(np.float32)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def preprocess(audio: np.ndarray, original_sr: int) -> tuple:
    """
    Full DSP preprocessing pipeline.
    Returns (spectrograms, bpm) ready for model inference.

    Steps:
        1.  Resample → 500 Hz
        2.  Spectral noise gate  (ambient noise suppression)
        3.  Bandpass filter      (20–500 Hz — matches model training)
        4.  Standardise          (mean=0, std=1)
        5.  BPM estimation       (Shannon Energy on 20–150 Hz band)
        6.  Split into 2 s chunks
        7.  Mel spectrogram per chunk  (128×128×1)
    """
    audio = resample_audio(audio, original_sr)         # 1
    audio = spectral_noise_gate(audio, TARGET_SAMPLE_RATE)  # 2
    audio = bandpass_filter(audio, TARGET_SAMPLE_RATE) # 3  (20–500 Hz)
    audio = standardize(audio)                         # 4

    bpm   = estimate_bpm_from_envelope(audio, TARGET_SAMPLE_RATE)  # 5

    chunks       = split_into_chunks(audio)            # 6
    spectrograms = [chunk_to_mel_spectrogram(c) for c in chunks]  # 7

    return spectrograms, bpm