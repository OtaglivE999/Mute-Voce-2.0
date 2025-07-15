# Muta Voce
Analyze audio for soft voices—far-field or similar—enhance the signal, perform voice-fingerprinting, and build a catalog of voice prints. Keep detailed logs of every detected voice, recording attributes that aid identification (voice color, formants, idiosyncrasies), plus date and precise timestamps. Provide an enhanced track that makes the voices clearly audible and distinct from background noise that other systems might misclassify due to limited training libraries

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
* Audio is captured and enhanced at **44.1 kHz** using **32‑bit float** WAV files.
* Fingerprinting now logs details like session ID, sample rate and spectral features to `logs/fingerprints.csv`.
