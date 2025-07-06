import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import soundfile as sf
import sys

LF_RANGE = (20, 100)
HF_RANGE = (20000, 24000)
OUTPUT_CSV = "lfn_analysis_results.csv"
SPECTROGRAM_FOLDER = "spectrograms"

os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
results = []

def convert_to_wav(input_path, output_path):
    command = ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1", output_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def analyze_audio(filepath, label, block_duration=None):
    """Analyze a single audio file.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    label : str
        Name used in results.
    block_duration : float, optional
        Duration of the blocks in seconds. When provided, the file is processed
        chunk by chunk which avoids loading the entire recording into memory.
    """

    with sf.SoundFile(filepath) as f:
        sr = f.samplerate
        block_frames = int(sr * block_duration) if block_duration else f.frames

        # Variables for tracking peaks across blocks
        max_lfn_db = -np.inf
        max_lfn_peak = 0
        max_hf_db = -np.inf
        max_hf_peak = 0

        spec_accum = None
        time_accum = []
        current_time = 0.0

        for block in f.blocks(blocksize=block_frames, dtype='float32'):
            if block.ndim > 1:
                block = block.mean(axis=1)

            freqs, times, Sxx = spectrogram(block, fs=sr, nperseg=4096, noverlap=2048)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            # LFN peak for this block
            lfn_mask = (freqs >= LF_RANGE[0]) & (freqs <= LF_RANGE[1])
            lfn_freqs = freqs[lfn_mask]
            lfn_spec = Sxx_db[lfn_mask, :]
            if lfn_spec.size:
                idx = np.argmax(lfn_spec)
                lfn_db_block = lfn_spec.flat[idx]
                lfn_peak_block = lfn_freqs[np.unravel_index(idx, lfn_spec.shape)[0]]
                if lfn_db_block > max_lfn_db:
                    max_lfn_db = lfn_db_block
                    max_lfn_peak = lfn_peak_block

            # Ultrasonic peak for this block
            hf_mask = (freqs >= HF_RANGE[0]) & (freqs <= HF_RANGE[1])
            hf_freqs = freqs[hf_mask]
            hf_spec = Sxx_db[hf_mask, :]
            if hf_freqs.size:
                idx = np.argmax(hf_spec)
                hf_db_block = hf_spec.flat[idx]
                hf_peak_block = hf_freqs[np.unravel_index(idx, hf_spec.shape)[0]]
                if hf_db_block > max_hf_db:
                    max_hf_db = hf_db_block
                    max_hf_peak = hf_peak_block

            # Accumulate spectrogram data up to 500 Hz
            freq_mask_500 = freqs <= 500
            spec_slice = Sxx_db[freq_mask_500, :]
            if spec_accum is None:
                spec_freqs = freqs[freq_mask_500]
                spec_accum = spec_slice
            else:
                spec_accum = np.hstack((spec_accum, spec_slice))

            time_accum.extend(times + current_time)
            current_time += len(block) / sr

    # Plot accumulated spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(time_accum, spec_freqs, spec_accum, shading='gouraud')
    plt.title(f"Spectrogram: {label}")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Intensity (dB)")
    out_img = os.path.join(SPECTROGRAM_FOLDER, f"{os.path.splitext(label)[0]}.png")
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()

    return {
        "Filename": label,
        "LFN Peak (Hz)": round(float(max_lfn_peak), 2),
        "LFN dB": round(float(max_lfn_db), 2),
        "Ultrasonic Peak (Hz)": round(float(max_hf_peak), 2),
        "Ultrasonic dB": round(float(max_hf_db), 2),
        "Spectrogram": out_img
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch analyze audio files for LFN and ultrasonic peaks")
    parser.add_argument("directory", help="Path to the audio directory")
    parser.add_argument("--block-duration", type=float, default=None,
                        help="Chunk size in seconds for processing large files")
    args = parser.parse_args()

    input_dir = args.directory
    block_duration = args.block_duration
    for file in os.listdir(input_dir):
        ext = file.lower().split('.')[-1]
        if ext in ["wav", "mp3", "mp4"]:
            full_path = os.path.join(input_dir, file)
            label = os.path.splitext(file)[0]
            wav_path = full_path
            if ext != "wav":
                wav_path = os.path.join(input_dir, f"{label}_converted.wav")
                convert_to_wav(full_path, wav_path)
            print(f"Analyzing {file}...")
            try:
                result = analyze_audio(wav_path, file, block_duration=block_duration)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {file}: {e}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(input_dir, OUTPUT_CSV)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Analysis complete. Results saved to {out_csv} and {SPECTROGRAM_FOLDER}/")

if __name__ == "__main__":
    main()
