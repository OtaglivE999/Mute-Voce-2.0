
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 1
BIT_DEPTH = 'FLOAT'

def find_input_device(name_substr=None):
    """Return an input device index.

    Parameters
    ----------
    name_substr : str, optional
        Substring to search for in device names. If provided, the first
        matching device with input channels is returned.
    """
    devices = sd.query_devices()

    if name_substr:
        for idx, d in enumerate(devices):
            if name_substr.lower() in d['name'].lower() and d['max_input_channels'] >= 1:
                return idx
        raise RuntimeError(f"Input device containing '{name_substr}' not found.")

    default = sd.default.device[0] if sd.default.device else None
    if default is not None and sd.query_devices(default)['max_input_channels'] > 0:
        return default

    for idx, d in enumerate(devices):
        if d['max_input_channels'] >= 1:
            return idx

    raise RuntimeError("No input device with recording channels available.")

def record_audio(duration_sec=10, device_index=None):
    """Record audio from the specified or default input device."""
    if device_index is None:
        device_index = find_input_device()
    print(f"[Recording] {duration_sec}s from device {device_index}...")

    audio = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='float32', device=device_index)
    sd.wait()
    return audio

def save_audio(filename, audio_data):
    sf.write(filename, audio_data.astype('float32'), SAMPLE_RATE, subtype=BIT_DEPTH)
    print(f"[Saved] Audio to {filename}")
