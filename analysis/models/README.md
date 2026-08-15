# Downloaded transcription models

This directory is a local cache, not an APC model registry.

`analysis/transcribe_videos.py` uses Faster-Whisper and downloads the selected
CTranslate2 model here on first use. For the default `distil-small.en` model,
the largest file is `model.bin`: approximately 317 MiB of neural-network
weights used to turn course audio into English text.

The cache is intentionally excluded from Git because it is:

- reproducibly downloadable from the declared upstream model identifier;
- unrelated to APC's poker perception/strategy checkpoints;
- large and platform/runtime specific;
- governed by its upstream model license and metadata.

Install the transcription environment from
`analysis/requirements-transcription.txt`, then run the transcription command.
Faster-Whisper will recreate the cache automatically.

The completed transcript evidence already committed to `analysis/transcripts/`
does not require this model at APC runtime.
