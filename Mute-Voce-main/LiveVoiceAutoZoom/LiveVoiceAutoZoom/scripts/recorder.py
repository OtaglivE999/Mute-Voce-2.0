"""Utility functions for recording audio from any connected microphone."""

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 44100
CHANNELS = 1
BIT_DEPTH = "FLOAT"


def list_input_devices():
    """Return a list ``[(index, name), ...]`` of available input devices."""
    devices = sd.query_devices()
    return [
        (idx, d["name"]) for idx, d in enumerate(devices) if d["max_input_channels"] >= 1
    ]


def find_input_device(preferred_name=None):
    """Return an input device index.

    If ``preferred_name`` is given, the first device whose name contains the
    provided string (case-insensitive) is returned. Otherwise the system default
    input device is used. If no default is configured, the first available input
    device is returned.
    """

    devices = sd.query_devices()

    if preferred_name:
        for idx, d in enumerate(devices):
            if preferred_name.lower() in d["name"].lower() and d["max_input_channels"] >= 1:
                return idx
        raise RuntimeError(f"Input device containing '{preferred_name}' not found.")

    default_idx = sd.default.device[0] if sd.default.device else None
    if default_idx is not None and sd.query_devices(default_idx)["max_input_channels"] >= 1:
        return default_idx

    for idx, d in enumerate(devices):
        if d["max_input_channels"] >= 1:
            return idx

    raise RuntimeError("No input device with at least one channel was found.")


def find_zoom_input():
    """Backward compatible helper for old scripts."""
    return find_input_device("Zoom")


def select_input_device(prompt="Select input device"):
    """Interactively ask the user to choose an input device.

    Returns the chosen device index, or raises ``RuntimeError`` if no devices
    are available.
    """

    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No input devices available")

    print(prompt)
    for idx, (dev_idx, name) in enumerate(devices):
        print(f"[{idx}] {name} (index {dev_idx})")

    choice = input("Enter number: ")
    try:
        return devices[int(choice)][0]
    except (ValueError, IndexError):
        raise RuntimeError("Invalid device selection")


def record_audio(duration_sec=10, device_index=None, device_name=None):
    """Record ``duration_sec`` seconds of audio from the chosen device."""

    if device_index is None:
        device_index = find_input_device(device_name)

    print(f"[Recording] {duration_sec}s from device {device_index}...")
    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    return audio


def save_audio(filename, audio_data):
    sf.write(filename, audio_data.astype("float32"), SAMPLE_RATE, subtype=BIT_DEPTH)
    print(f"[Saved] Audio to {filename}")


if __name__ == "__main__":
    # Allow quick inspection of available input devices when run directly.
    for idx, name in list_input_devices():
        print(f"[{idx}] {name}")

