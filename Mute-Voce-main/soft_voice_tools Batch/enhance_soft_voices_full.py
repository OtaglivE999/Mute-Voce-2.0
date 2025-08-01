import os
import sys
import subprocess
import librosa
import soundfile as sf
import numpy as np

print("🔊 Full Soft Voice Enhancer + Transcriber")

# allow passing the input file on the command line
if len(sys.argv) > 1:
    input_path = sys.argv[1]
else:
    input_path = input(
        "Enter full path to your audio/video file (.wav, .mp3, .mp4): "
    ).strip().strip('"')
def sanitize_path(path: str) -> str:
    """Try to resolve problematic paths that contain spaces.

    Only the file name is renamed if necessary. Parent directories are left
    untouched to avoid permission issues. If an underscored file already
    exists, that version is used.
    """

    if " " not in path:
        return path

    directory, filename = os.path.split(path)
    underscored = filename.replace(" ", "_")
    candidate = os.path.join(directory, underscored)

    if os.path.exists(candidate):
        print(f"ℹ️ Using existing path '{candidate}'")
        return candidate

    if os.path.exists(path) and filename != underscored:
        try:
            os.rename(path, candidate)
            print(f"ℹ️ Renamed '{path}' to '{candidate}'")
            return candidate
        except Exception as e:
            print(f"⚠️ Could not rename '{path}': {e}")

    return path

input_path = sanitize_path(input_path)

if not os.path.exists(input_path):
    print(f"❌ File not found: {input_path}")
    sys.exit(1)
if not os.path.isfile(input_path):
    print(f"❌ Path is not a file: {input_path}")
    sys.exit(1)

base_name = os.path.splitext(os.path.basename(input_path))[0]
ext = os.path.splitext(input_path)[1].lower()

try:
    print("📥 Loading audio...")
    y, sr = librosa.load(input_path, sr=None)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    sys.exit(1)

try:
    print("🎚️ Enhancing soft voices...")
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    gain_mask = np.where(rms_db < -30, 10**(((-30 - rms_db) / 20)), 1.0)
    gain_expanded = np.repeat(gain_mask, hop_length)
    gain_expanded = gain_expanded[:len(y)] if len(gain_expanded) > len(y) else np.pad(gain_expanded, (0, len(y) - len(gain_expanded)))
    y_enhanced = np.clip(y * gain_expanded, -1.0, 1.0)
    wav_output = f"enhanced_{base_name}.wav"
    sf.write(wav_output, y_enhanced, sr)
    print(f"✅ Saved enhanced audio: {wav_output}")
except Exception as e:
    print(f"❌ Enhancement error: {e}")
    sys.exit(1)

# Convert back to MP4 if needed
if ext == ".mp4":
    try:
        mp4_output = f"enhanced_{base_name}.mp4"
        print("🎞️ Rebuilding MP4 with enhanced audio...")
        # build ffmpeg command as a list to avoid shell injection
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-i",
            wav_output,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            mp4_output,
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Rebuilt MP4 saved as: {mp4_output}")
    except Exception as e:
        print(f"⚠️ Could not rebuild MP4: {e}")

# Whisper transcription
try:
    print("📝 Transcribing using Whisper (if installed)...")
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(wav_output)
    with open(f"transcript_{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"✅ Transcript saved: transcript_{base_name}.txt")
except Exception as e:
    print(f"ℹ️ Whisper transcription skipped or failed: {e}")
