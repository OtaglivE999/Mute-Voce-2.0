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
