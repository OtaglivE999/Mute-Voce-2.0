# Muta Voce
Analyze audio for soft voices—far-field or similar—enhance the signal, perform voice-fingerprinting, and build a catalog of voice prints. Keep detailed logs of every detected voice, recording attributes that aid identification (voice color, formants, idiosyncrasies), plus date and precise timestamps. Provide an enhanced track that makes the voices clearly audible and distinct from background noise that other systems might misclassify due to limited training libraries


## Usage

Run `run_all.sh` (Linux/macOS) or `run_all.bat` (Windows). These scripts install
the Python packages listed in `requirements.txt` and then invoke the enhancer.
Provide a path to an audio or video file either as a command-line argument or
when prompted.
If the path contains spaces, wrap it in quotes. The input must be a file, not a
folder. Example:

```sh
./run_all.sh "path/to/My File.mp4"
```

If no path is given, the script prompts for one.

The optional transcription step relies on the `openai-whisper` library, which
requires PyTorch and is not included in `requirements.txt`. Install it manually
with `pip install openai-whisper` if you want transcripts.

If you accidentally provide a path with spaces that cannot be resolved, the
program attempts to replace spaces in the **file name** with underscores. This
does not rename parent folders (which could require special permissions). If an
underscored file already exists or the original can be renamed safely, that
version is used automatically. Otherwise you will be prompted that the path is
invalid.


 main
## LiveVoiceAutoZoom

The LiveVoiceAutoZoom tool can now work with any connected microphone. When launching the GUI you will be presented with a list of available input devices and may select the one you wish to record from.

For the command line script ``live_zoom_record_and_analyze.py`` you can list
devices with ``--list-devices`` or interactively choose one with
``--choose-device``. Alternatively specify ``--device-index`` or ``--device-name``
to select a microphone directly.

### Dependencies

These tools require the `sounddevice` and `soundfile` Python packages.
 
On Linux, the system PortAudio library is also needed. Install the
dependencies with:

```bash
pip install -r requirements.txt
sudo apt-get install libportaudio2  # Linux only
python - <<'PY'
import sounddevice
print('sounddevice loaded', sounddevice.__version__)
PY
```

=======
Install them with:

```bash
pip install -r requirements.txt
```


## LFN Batch File Analyzer

The script `LFN_Docker_Toolkit_Extended/LFN_Docker_Toolkit_Extended/lfn_batch_file_analyzer.py`
analyzes audio files for low‑frequency noise (LFN) and ultrasonic peaks.

```
python lfn_batch_file_analyzer.py <directory> [--block-duration SECONDS]
```

Use `--block-duration` to set the chunk size when reading files. Processing
recordings block by block prevents memory errors with very long audio while still
producing a spectrogram of the entire file (only frequencies up to 500 Hz are
stored).

## Live Zoom Recorder

`LiveVoiceAutoZoom/scripts/live_zoom_record_and_analyze.py` now streams audio
directly to disk instead of holding the full recording in memory. Use the
optional `block_duration` parameter of `record_audio()` to control how much
audio is buffered at a time (default is 10 seconds). This reduces memory usage
when capturing long sessions.

The recorder works with any accessible microphone. By default the script uses
the system's default input device. Pass `--device` to specify a device index or
name substring.

### Command-line options

```
python live_zoom_record_and_analyze.py [--device DEVICE] [--duration SECS] [--block-duration SECS]
```

* `--device` – input device index or name substring (default: system default)
* `--duration` – recording length in seconds (default: 4440)
* `--block-duration` – length of audio chunks written to disk (default: 10)

On Windows, use `run_zoom.bat` to launch the recorder. Any arguments passed to the
batch file are forwarded to the Python script:

```
run_zoom.bat --device "USB Microphone"
```

* Audio is captured and enhanced at **48 kHz** using **32‑bit float** WAV files.
* Fingerprinting now logs details like session ID, sample rate and spectral features to `logs/fingerprints.csv`.
 main



