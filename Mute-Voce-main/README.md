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

If you accidentally provide a path with spaces that cannot be resolved, the
program attempts to replace spaces in the **file name** with underscores. This
does not rename parent folders (which could require special permissions). If an
underscored file already exists or the original can be renamed safely, that
version is used automatically. Otherwise you will be prompted that the path is
invalid.
