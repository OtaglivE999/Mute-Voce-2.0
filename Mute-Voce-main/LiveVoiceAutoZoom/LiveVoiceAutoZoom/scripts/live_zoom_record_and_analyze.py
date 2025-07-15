import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
import argparse

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

RECORD_SECONDS = 74 * 60  # 4440 seconds
SAMPLE_RATE = 44100
CHANNELS = 1
SESSION_ID = time.strftime("%Y%m%d_%H%M%S")
RECORD_PATH = f"recordings/session_{SESSION_ID}.wav"
ENHANCE_PATH = f"enhanced/session_{SESSION_ID}_enhanced.wav"
FINGERPRINT_PATH = f"fingerprints/voiceprint_{SESSION_ID}.npy"
LOG_PATH = f"logs/session_{SESSION_ID}.log"
CSV_LOG_PATH = "logs/fingerprints.csv"

def record_audio(filename, duration, samplerate, channels, device_idx, block_duration=10):
    """Record audio directly to disk to avoid large memory use.

    Parameters
    ----------
    filename : str
        Path to output WAV file.
    duration : int or float
        Recording duration in seconds.
    samplerate : int
        Sample rate to record at.
    channels : int
        Number of channels to record.
    device_idx : int
        Index of the input device.
    block_duration : float, optional
        Duration of blocks written to disk in seconds.
    """

    print(f"[+] Recording {duration}s from device {device_idx}...")
    block_frames = int(samplerate * block_duration)

    with sf.SoundFile(filename, mode="w", samplerate=samplerate, channels=channels, subtype="FLOAT") as f:
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[WARNING] Input status: {status}")
            f.write(indata.copy())

        with sd.InputStream(samplerate=samplerate,
                            channels=channels,
                            device=device_idx,
                            dtype='float32',
                            blocksize=block_frames,
                            callback=callback):
            sd.sleep(int(duration * 1000))

    print(f"[+] Saved to {filename}")
    return filename

def enhance_audio(input_file, output_file):
    data, sr = sf.read(input_file)
    from scipy.signal import butter, lfilter
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    filtered_data = bandpass_filter(data, 150, 7000, sr)
    sf.write(output_file, filtered_data.astype('float32'), sr, subtype='FLOAT')
    print(f"[+] Enhanced audio saved to {output_file}")
    return output_file

def fingerprint_audio(file_path):
    wav = preprocess_wav(file_path)
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    np.save(FINGERPRINT_PATH, embed)
    print(f"[+] Fingerprint saved to {FINGERPRINT_PATH}")

    try:
        import csv
        import librosa
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=SAMPLE_RATE)
        mean_f0 = float(np.nanmean(f0))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)))
        row = {
            "session_id": SESSION_ID,
            "fingerprint_file": os.path.basename(FINGERPRINT_PATH),
            "samplerate": SAMPLE_RATE,
            "format": "float32",
            "mean_freq": round(mean_f0, 2),
            "voice_color": round(centroid, 2)
        }
        exists = os.path.exists(CSV_LOG_PATH)
        with open(CSV_LOG_PATH, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[WARNING] Failed to log fingerprint details: {e}")

    return embed

def compare_with_existing(embed):
    match_log = []
    for f in os.listdir("fingerprints"):
        if f.endswith(".npy") and SESSION_ID not in f:
            ref = np.load(os.path.join("fingerprints", f))
            similarity = np.inner(embed, ref)
            if similarity > 0.75:
                match_log.append(f"Match with {f}: similarity {similarity:.2f}")
    return match_log

if __name__ == "__main__":
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("enhanced", exist_ok=True)
    os.makedirs("fingerprints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    parser = argparse.ArgumentParser(description="Record and analyze audio from a microphone")
    parser.add_argument("--device", help="Input device index or name substring", default=None)
    parser.add_argument("--duration", type=float, default=RECORD_SECONDS, help="Recording length in seconds")
    parser.add_argument("--block-duration", type=float, default=10.0, help="Duration of blocks written to disk")
    args = parser.parse_args()

    try:
        if args.device is not None:
            try:
                device_index = int(args.device)
            except ValueError:
                device_index = find_input_device(args.device)
        else:
            device_index = find_input_device()

        record_audio(RECORD_PATH, args.duration, SAMPLE_RATE, CHANNELS, device_index, args.block_duration)
        enhance_audio(RECORD_PATH, ENHANCE_PATH)
        embedding = fingerprint_audio(ENHANCE_PATH)
        matches = compare_with_existing(embedding)

        with open(LOG_PATH, "w") as f:
            f.write(f"Session: {SESSION_ID}\n")
            f.write("\n".join(matches))
            print(f"[+] Log written to {LOG_PATH}")
    except Exception as e:
        print(f"[ERROR] {e}")
