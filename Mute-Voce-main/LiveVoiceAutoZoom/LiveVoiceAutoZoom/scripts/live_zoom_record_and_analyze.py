import os
import sys
import time
import argparse
import csv
import warnings
import numpy as np
import sounddevice as sd
import soundfile as sf

# Suppress pkg_resources deprecation warnings triggered by webrtcvad
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

from resemblyzer import VoiceEncoder, preprocess_wav

from recorder import find_input_device, list_input_devices, select_input_device
from speaker_recognition import cluster_unknown_embeddings
from vad_enhancer import detect_voiced, enhance_audio as vad_enhance


RECORD_SECONDS = 74 * 60  # 4440 seconds
SAMPLE_RATE = 44100
CHANNELS = 1
SESSION_ID = time.strftime("%Y%m%d_%H%M%S")
RECORD_PATH = f"recordings/session_{SESSION_ID}.wav"
ENHANCE_PATH = f"enhanced/session_{SESSION_ID}_enhanced.wav"
FINGERPRINT_PATH = f"fingerprints/voiceprint_{SESSION_ID}.npy"
LOG_PATH = f"logs/session_{SESSION_ID}.log"
CSV_LOG_PATH = "logs/fingerprints.csv"
SPEAKER_SUMMARY_CSV = "logs/speaker_summary.csv"


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

    print(f"[+] Recording {duration}s from Zoom H6 (Device {device_idx})...")
    block_frames = int(samplerate * block_duration)

    with sf.SoundFile(
        filename, mode="w", samplerate=samplerate, channels=channels, subtype="FLOAT"
    ) as f:

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[WARNING] Input status: {status}")
            f.write(indata.copy())

        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            device=device_idx,
            dtype="float32",
            blocksize=block_frames,
            callback=callback,
        ):
            sd.sleep(int(duration * 1000))

    print(f"[+] Saved to {filename}")


def enhance_audio(input_file, output_file):
    """Apply VAD-based enhancement with mid-range boost."""
    data, sr = sf.read(input_file)
    voiced = detect_voiced(data, sr)
    enhanced = vad_enhance(voiced, sr)
    sf.write(output_file, enhanced.astype("float32"), sr, subtype="FLOAT")
    print(f"[+] Enhanced audio saved to {output_file}")
    return output_file


def fingerprint_audio(file_path):
    """Create embeddings and log per-speaker fingerprints."""
    wav = preprocess_wav(file_path)
    encoder = VoiceEncoder()
    embed, partials, _ = encoder.embed_utterance(wav, return_partials=True)
    np.save(FINGERPRINT_PATH, embed)
    print(f"[+] Fingerprint saved to {FINGERPRINT_PATH}")

    try:
        import librosa

        y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=SAMPLE_RATE)
        mean_f0 = float(np.nanmean(f0))
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE))
        )
        row = {
            "session_id": SESSION_ID,
            "fingerprint_file": os.path.basename(FINGERPRINT_PATH),
            "samplerate": SAMPLE_RATE,
            "format": "float32",
            "mean_freq": round(mean_f0, 2),
            "voice_color": round(centroid, 2),
        }
        exists = os.path.exists(CSV_LOG_PATH)
        with open(CSV_LOG_PATH, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[WARNING] Failed to log fingerprint details: {e}")

    n_clusters = min(5, len(partials)) or 1
    labels = cluster_unknown_embeddings(partials, n_clusters=n_clusters)
    speaker_files = []
    for idx in sorted(set(labels)):
        speaker_embedding = partials[labels == idx].mean(axis=0)
        out_file = f"fingerprints/voiceprint_{SESSION_ID}_speaker{idx + 1}.npy"
        np.save(out_file, speaker_embedding)
        speaker_files.append(os.path.basename(out_file))

    exists = os.path.exists(SPEAKER_SUMMARY_CSV)
    with open(SPEAKER_SUMMARY_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not exists:
            writer.writerow(["session_id", "num_speakers", "speaker_files"])
        writer.writerow([SESSION_ID, len(set(labels)), ";".join(speaker_files)])

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
    parser = argparse.ArgumentParser(
        description="Record and analyze audio from a microphone"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    parser.add_argument("--device-index", type=int, help="Input device index to use")
    parser.add_argument("--device-name", help="Search for input device by name")
    parser.add_argument(
        "--choose-device",
        action="store_true",
        help="Interactively choose an input device",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=RECORD_SECONDS,
        help="Recording length in seconds",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=10.0,
        help="Duration of blocks written to disk",
    )
    args = parser.parse_args()

    if args.list_devices:
        for idx, name in list_input_devices():
            print(f"{idx}: {name}")
        sys.exit(0)

    os.makedirs("recordings", exist_ok=True)
    os.makedirs("enhanced", exist_ok=True)
    os.makedirs("fingerprints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        if args.choose_device:
            device_index = select_input_device()
        elif args.device_index is not None:
            device_index = args.device_index
        else:
            device_index = find_input_device(args.device_name)

        record_audio(
            RECORD_PATH,
            args.duration,
            SAMPLE_RATE,
            CHANNELS,
            device_index,
            args.block_duration,
        )
        enhance_audio(RECORD_PATH, ENHANCE_PATH)
        embedding = fingerprint_audio(ENHANCE_PATH)
        matches = compare_with_existing(embedding)

        with open(LOG_PATH, "w") as f:
            f.write(f"Session: {SESSION_ID}\n")
            f.write("\n".join(matches))
            print(f"[+] Log written to {LOG_PATH}")
    except Exception as e:
        print(f"[ERROR] {e}")

