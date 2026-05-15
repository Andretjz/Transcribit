# 🎙️ Transcribit

**Audio Intelligence Platform** — Transcripción, diarización de hablantes e inteligencia contextual para audio en múltiples idiomas.

---

## ✨ Funcionalidades

- 🗣️ **Transcripción** con WhisperX (large-v3-turbo) con alineación de palabras
- 👥 **Diarización de hablantes** con pyannote.audio 4.x (quién habla cuándo)
- 🧠 **Identificación de nombres** automática por patrones de lenguaje y embeddings acústicos
- 🎭 **Análisis de tono/emoción** mediante SpeechBrain (wav2vec2-IEMOCAP)
- 🌍 **Multilingüe**: alemán, inglés, español, francés, italiano
- 🔍 **Análisis profundo con LLM** (Ollama / llama3.1:8b) según contexto:
  - `interview` → pares Q&A, desviaciones, velocidad de reacción
  - `language_practice` / `self_recorded` → fluencia, gramática, alternativas nativas
  - `lecture` → segmentación temática, análisis de entrega
- 🇩🇪🇬🇧 **Detector de Denglisch** (mezcla código alemán-inglés)
- ❄️ **Cold-start acústico** para los primeros 60s sin diarización estable

---

## 🛠️ Requisitos

- Python 3.10+
- CUDA 12.6 + GPU NVIDIA (recomendado)
- [Ollama](https://ollama.ai) con `llama3.1:8b` instalado
- Token de HuggingFace con acceso a los modelos `pyannote` ([solicitar aquí](https://huggingface.co/pyannote/speaker-diarization-3.1))

---

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/Transcribit.git
cd Transcribit

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Instalar PyTorch con CUDA (primero)
pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 4. Instalar el resto de dependencias
pip install -r requirements.txt

# 5. Configurar token de HuggingFace
# Crear fichero .env con:
#   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## ▶️ Uso

```bash
# Iniciar el servidor backend
uvicorn app:app --host 0.0.0.0 --port 8000

# Abrir el frontend
# Abrir index.html en el navegador o servir con:
python -m http.server 3000
```

El frontend estará en `http://localhost:3000` y la API en `http://localhost:8000`.

---

## 📡 API

| Endpoint | Método | Descripción |
|---|---|---|
| `/process` | `POST` | Procesa un fichero de audio y devuelve transcripción + análisis |
| `/health` | `GET` | Estado del servidor |

### Parámetros de `/process`
- `file` — fichero de audio (wav, mp3, m4a, ogg…)
- `language` — `de`, `en`, `es`, `fr`, `it`
- `context` — `interview`, `lecture`, `language_practice`, `self_recorded`, `unknown`

---

## 🏗️ Arquitectura

```
index.html          ← SPA frontend (vanilla JS)
app.py              ← FastAPI backend, pipeline principal
custom_interface.py ← Adaptador SpeechBrain personalizado
requirements.txt    ← Dependencias Python
```

Los modelos de IA se descargan automáticamente en el primer arranque (wav2vec2, pyannote, whisper).

---

## 📄 Licencia

Ver [LICENSE](LICENSE)
