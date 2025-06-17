# Muta Voce
Analyze audio for soft voices—far-field or similar—enhance the signal, perform voice-fingerprinting, and build a catalog of voice prints. Keep detailed logs of every detected voice, recording attributes that aid identification (voice color, formants, idiosyncrasies), plus date and precise timestamps. Provide an enhanced track that makes the voices clearly audible and distinct from background noise that other systems might misclassify due to limited training libraries

## LiveVoiceAutoZoom

The LiveVoiceAutoZoom tool can now work with any connected microphone. When launching the GUI you will be presented with a list of available input devices and may select the one you wish to record from.

For the command line script ``live_zoom_record_and_analyze.py`` you can list
devices with ``--list-devices`` or interactively choose one with
``--choose-device``. Alternatively specify ``--device-index`` or ``--device-name``
to select a microphone directly.

### Dependencies

These tools require the `sounddevice` and `soundfile` Python packages.
Install them with:

```bash
pip install -r requirements.txt
```

