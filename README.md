# 🎙️ Transcribit

**Audio Intelligence Platform** — Multi-speaker transcription, speaker diarization, and context-aware audio intelligence for multilingual recordings.

---

## ✨ Features

- 🗣️ **Transcription** with WhisperX (large-v3-turbo) and word-level alignment
- 👥 **Speaker diarization** via pyannote.audio 4.x (who speaks when)
- 🧠 **Automatic speaker name identification** from linguistic patterns and acoustic embeddings
- 🎭 **Tone / emotion analysis** using SpeechBrain (wav2vec2-IEMOCAP)
- 🌍 **Multilingual**: German, English, Spanish, French, Italian
- 🔍 **LLM-powered deep analysis** (Ollama / llama3.1:8b) by audio context:
  - `interview` → Q&A pairs, topic deviation, reaction time, answer quality scoring
  - `language_practice` / `self_recorded` → fluency, grammar corrections, native-like alternatives
  - `lecture` → topic segmentation, delivery analysis
  - `unknown` → general transcript intelligence
- 🇩🇪🇬🇧 **Denglisch detector** — tracks English code-switching in German speech
- ❄️ **Acoustic cold-start** for the first 60 s before stable diarization is available

---

## 🛠️ Requirements

- Python 3.10+
- CUDA 12.6 + NVIDIA GPU (strongly recommended)
- [Ollama](https://ollama.ai) running locally with `llama3.1:8b` pulled
- HuggingFace account token with access to the pyannote models ([request access here](https://huggingface.co/pyannote/speaker-diarization-3.1))

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Andretjz/Transcribit.git
cd Transcribit

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install PyTorch with CUDA support first
pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> [!NOTE]
> The pyannote and SpeechBrain models are downloaded automatically on first run and cached locally (not included in the repository).

---

## ▶️ Running

```bash
# Start the FastAPI backend
uvicorn app:app --host 0.0.0.0 --port 8000

# Serve the frontend (open in browser)
python -m http.server 3000
# Then open: http://localhost:3000
```

When prompted, paste your HuggingFace token — it is sent per-request and never stored on disk.

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/process` | `POST` | Process an audio file and return full transcript + analysis |
| `/health` | `GET` | Server and GPU status |

### `/process` — form parameters

| Parameter | Type | Description |
|---|---|---|
| `audio` | file | Audio file (wav, mp3, m4a, ogg, …) |
| `hf_token` | string | HuggingFace token for pyannote models |
| `language` | string | `de`, `en`, `es`, `fr`, `it` |
| `num_speakers` | int | Known number of speakers (0 = auto-detect) |
| `context` | string | `interview`, `lecture`, `language_practice`, `self_recorded`, `unknown` |

---

## 🏗️ Architecture

```
index.html           ← Single-page frontend (vanilla JS)
app.py               ← FastAPI backend — full processing pipeline
custom_interface.py  ← Custom SpeechBrain adapter for wav2vec2 emotion model
requirements.txt     ← Python dependencies
```

**Processing pipeline (6 passes):**

1. **Pass 1** — WhisperX transcription + word alignment
2. **Pass 2** — pyannote diarization + speaker embedding fingerprints
3. **Pass 3** — Acoustic cold-start (silence gating + pitch change-points)
4. **Pass 4a** — Regex-based name resolution from self-introduction patterns
5. **Pass 4b** — LLM role inference + speaker name correction (2-stage)
6. **Pass 5** — Voice blueprints + emotion/tone classification + Denglisch detection

---

## 📄 License

See [LICENSE](LICENSE)
