
import numpy as np
import warnings

# Suppress deprecation warnings emitted by webrtcvad's use of pkg_resources
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import webrtcvad
from scipy.signal import butter, lfilter

SAMPLE_RATE = 44100
FRAME_DURATION = 30  # ms
VAD_MODE = 2  # 0-3: higher = more aggressive

vad = webrtcvad.Vad(VAD_MODE)

def frame_generator(audio, sample_rate, frame_duration_ms):
    frame_len = int(sample_rate * frame_duration_ms / 1000)
    for i in range(0, len(audio) - frame_len + 1, frame_len):
        yield audio[i:i + frame_len]

def detect_voiced(audio, sample_rate=SAMPLE_RATE):
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    frames = frame_generator(audio, sample_rate, FRAME_DURATION)
    voiced = []
    for frame in frames:
        if vad.is_speech(frame.tobytes(), sample_rate):
            voiced.extend(frame)
    return np.array(voiced, dtype=np.int16)

def apply_midrange_enhancement(audio, sample_rate=SAMPLE_RATE, low=300, high=3000, gain=1.5):
    """Boost mid-range frequencies to improve intelligibility."""
    nyq = 0.5 * sample_rate
    low /= nyq
    high /= nyq
    b, a = butter(2, [low, high], btype="band")
    mid = lfilter(b, a, audio)
    enhanced = audio + gain * mid
    return np.clip(enhanced, -1.0, 1.0)


def enhance_audio(voiced_audio, sample_rate=SAMPLE_RATE):
    """Normalize, amplify, and boost mid-range frequencies."""
    if len(voiced_audio) == 0:
        return np.zeros(1, dtype=np.float32)
    if voiced_audio.dtype != np.float32:
        voiced_audio = voiced_audio.astype(np.float32) / 32767
    normalized = voiced_audio / np.max(np.abs(voiced_audio))
    amplified = normalized * 0.9
    mid_boosted = apply_midrange_enhancement(amplified, sample_rate)
    return mid_boosted.astype(np.float32)
