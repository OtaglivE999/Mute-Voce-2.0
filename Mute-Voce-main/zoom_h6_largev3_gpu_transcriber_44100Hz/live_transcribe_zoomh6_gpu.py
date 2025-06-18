import argparse
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import traceback
from faster_whisper import WhisperModel

# ------------------ CONFIGURATION ------------------
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_DURATION = 5
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float16"
DEVICE_TYPE = "cuda"

# ------------------ INITIALIZATION ------------------
audio_queue = queue.Queue()
model = None

def list_input_devices():
    devices = sd.query_devices()
    return [
        (idx, d["name"]) for idx, d in enumerate(devices) if d["max_input_channels"] >= CHANNELS
    ]

def find_input_device(name=None):
    devices = sd.query_devices()
    if name:
        for i, d in enumerate(devices):
            if name.lower() in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
                return i
    default_idx = sd.default.device[0]
    if default_idx is not None and default_idx >= 0:
        if devices[default_idx]["max_input_channels"] >= CHANNELS:
            return default_idx
    for i, d in enumerate(devices):
        if d["max_input_channels"] >= CHANNELS:
            return i
    raise RuntimeError("No suitable input device found")

def select_input_device(prompt="Select input device"):
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No input devices available")
    print(prompt)
    for n, (idx, name) in enumerate(devices):
        print(f"[{n}] {name} (index {idx})")
    choice = input("Enter number: ")
    try:
        return devices[int(choice)][0]
    except (ValueError, IndexError):
        raise RuntimeError("Invalid device selection")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARNING] Input stream status: {status}")
    if indata.shape[0] == 0:
        print("[WARNING] No audio data received")
        return
    audio_queue.put(indata.copy())

def transcribe_audio(audio_data):
    try:
        if np.max(np.abs(audio_data)) < 0.01:
            print("[INFO] Skipped silent audio chunk")
            return
        print("[INFO] Transcribing...")
        segments, _ = model.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )
        for segment in segments:
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
    except Exception as e:
        print(f"[ERROR] Transcription failed:\n{traceback.format_exc()}")

def capture_audio(device_index):
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            callback=audio_callback,
                            blocksize=int(SAMPLE_RATE * 0.5),
                            dtype='float32',
                            device=device_index):
            print("[INFO] Audio stream started. Listening...")
            buffer = np.empty((0, CHANNELS), dtype='float32')
            last_time = time.time()
            while True:
                data = audio_queue.get()
                buffer = np.vstack((buffer, data))
                if time.time() - last_time >= CHUNK_DURATION:
                    chunk = buffer[:, 0]
                    threading.Thread(target=transcribe_audio, args=(chunk.copy(),)).start()
                    buffer = np.empty((0, CHANNELS), dtype='float32')
                    last_time = time.time()
    except Exception as e:
        print(f"[FATAL] Failed to open audio stream:\n{traceback.format_exc()}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live transcription from microphone")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    parser.add_argument("--device-index", type=int, help="Input device index")
    parser.add_argument("--device-name", help="Search for input device by name")
    parser.add_argument("--choose-device", action="store_true", help="Interactively choose an input device")
    args = parser.parse_args()

    if args.list_devices:
        for idx, name in list_input_devices():
            print(f"{idx}: {name}")
        raise SystemExit

    print("[INFO] Initializing Whisper model...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[FATAL] Whisper model failed to load:\n{traceback.format_exc()}")
        exit(1)

    try:
        if args.choose_device:
            device_index = select_input_device()
        elif args.device_index is not None:
            device_index = args.device_index
        else:
            device_index = find_input_device(args.device_name)
        capture_audio(device_index)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    except Exception as e:
        print(f"[FATAL] Unhandled error:\n{traceback.format_exc()}")
