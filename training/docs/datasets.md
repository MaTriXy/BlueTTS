# Training data sources

This project was trained on different public corpora depending on language.

| Language | Dataset | Notes |
|----------|---------|--------|
| Hebrew | [SententicDataTTS](https://huggingface.co/datasets/notmax123/SententicDataTTS) | Hebrew and English utterances; male and female speakers; 44.1 kHz, time-stretched (slowed). See below. |
| English | [LibriTTS](https://www.openslr.org/60/) | Full **SLR60** distribution: all official train/dev/test archives (~585 h @ 24 kHz). See below. |
| German, Spanish, Italian | [M-AILABS Speech Dataset](https://github.com/i-celeste-aurora/m-ailabs-dataset) | LJSpeech-style layout per book/voice (`metadata.csv`, `wavs/`); mono WAV at 16 kHz. |

---

## Hebrew: SententicDataTTS

**Hub page:** [notmax123/SententicDataTTS](https://huggingface.co/datasets/notmax123/SententicDataTTS)

A Hebrew and English TTS dataset with male and female speakers, resampled to 44.1 kHz and time-stretched (slowed).

### Audio archives

- **slow_44K.7z** — audio generated with Chatterbox; WAV at 44.1 kHz, slowed.
- **Mamre_generated.7z** — audio generated with MamreTTS.

---

## English: LibriTTS

**OpenSLR:** [LibriTTS (SLR60)](https://www.openslr.org/60/) — multi-speaker English TTS audio at **24 kHz**, derived from the LibriSpeech source materials, with sentence-level cuts and original plus normalized text.

Training used the **entire corpus as published** on that page: all listed archives, i.e. **`train-clean-100`**, **`train-clean-360`**, **`train-other-500`**, **`dev-clean`**, **`dev-other`**, **`test-clean`**, and **`test-other`** (see the download list on [openslr.org/60](https://www.openslr.org/60/)). License on the resource page: **CC BY 4.0**.

---

## German, Spanish, Italian: M-AILABS

**Documentation mirror / overview:** [i-celeste-aurora/m-ailabs-dataset](https://github.com/i-celeste-aurora/m-ailabs-dataset)

Per-language packs (e.g. `de_DE`, `es_ES`, `it_IT`) use a book/voice hierarchy with `metadata.csv` (and optionally `metadata_mls.json`) next to a **`wavs/`** directory. Audio is mono **16 kHz** WAV in the distribution described on that page. Format is compatible with common LJSpeech-style preprocessing (pipe-separated text fields in `metadata.csv`).

**License:** see the copyright and redistribution terms in the [M-AILABS dataset README](https://github.com/i-celeste-aurora/m-ailabs-dataset/blob/master/README.md).
