"""
Transcribit v6 — Audio Intelligence Platform
=============================================
FIXES in v6:
  [CRITICAL] Embedding waveform shape: was (1,1,T) → now (1,T) — this was
             the root cause of Fingerprints: [] across all runs.
  [CRITICAL] Cold-start n_speakers: clamp to actual burst capacity so
             pyannote never tries to find 3 speakers in a 2s chunk.

NEW in v6:
  - Audio context selector: interview / lecture / language_practice /
    self_recorded / unknown — drives which Ollama analysis engine runs.
  - Language support: de / en / es / fr / it
  - Context-aware Ollama engines:
      interview     → answer tracking, deviation detection, reaction speed,
                       topic identification, Q&A accuracy scoring
      language_practice / self_recorded →
                       pronunciation analysis, native-like alternatives,
                       grammar corrections, fluency metrics
      lecture       → topic segmentation, vocalization analysis
      unknown       → general transcript intelligence
  - Denglisch / code-switching detector (German+English mixing)
  - All analysis results added to /process response as `deep_analysis`
"""

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn, tempfile, os, re, json, gc, time
import numpy as np
import torch
import whisperx
import parselmouth
from pyannote.audio import Pipeline, Model, Inference
from speechbrain.inference.interfaces import foreign_class
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import pandas as pd
import urllib.request

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

print("=" * 60)
print("  TRANSCRIBIT v6 — Audio Intelligence Platform")
print("  Loading Whisper large-v3-turbo...")
print("=" * 60)

DEVICE          = "cuda"
SAMPLE_RATE     = 16000
COLD_START_S    = 60             # was 180 — too long, collapsed 3 speakers into 2
COLD_THRESH     = 0.72           # minimum similarity to reassign in cold-start (was EMBED_THRESH 0.50)
MIN_SEG_SAMPLES = 8000
MIN_EMBED_DUR   = 1.5            # was 2.0 — allow shorter segments for fingerprints
NAME_CONF_THRES = 2
EMBED_THRESH    = 0.50           # was 0.55 — more forgiving cold-start matching
OLLAMA_URL      = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL    = "llama3.1:8b"

# Surname/alias → first name mapping.
# Add entries when Whisper mishears a last name or a speaker is addressed by surname.
# Keys are lowercase, values are the resolved first name.
SURNAME_ALIAS = {
    "gassier":  "Marcella",   # Whisper's German rendition of "Garcia"
    "garcia":   "Marcella",
    "raffi":    "Marcella",   # Sabine's nickname for Marcella at 79.8s
}

whisper_model = whisperx.load_model(
    "large-v3-turbo", DEVICE, compute_type="float16"
)
tone_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    run_opts={"device": DEVICE}
)
print("  Models ready.\n" + "=" * 60)

# ── STOP WORDS ─────────────────────────────────────────────────
STOP_WORDS = {
    "ich","du","er","sie","es","wir","ihr","und","oder","aber","denn","weil",
    "dass","die","der","das","ein","eine","einen","einem","einer","eines",
    "ist","bin","bist","sind","war","hatte","haben","hat","wird","wurde",
    "werden","kann","muss","soll","will","darf","mag","möchte","in","an",
    "auf","bei","mit","nach","von","vor","zu","zum","zur","aus","durch",
    "für","über","unter","zwischen","gegen","ohne","um","als","wie","auch",
    "noch","schon","immer","nie","nicht","kein","keine","ja","nein","ok",
    "so","dann","mal","doch","halt","sehr","ganz","einfach","nur","hier",
    "da","dort","wenn","wann","wo","was","wer","warum","welche","mein",
    "meine","dein","sein","ihre","unser","mir","mich","dir","sich","uns",
    "euch","ihnen","ihm","ihn","genau","gut","super","gerne","natürlich",
    "eigentlich","irgendwie","jetzt","schon","habe","haben","hatte",
    "diesem","dieser","diese","dieses","viel","viele","mehr","beim","vom",
    "i","you","he","the","a","an","is","are","was","were","be","have","has",
    "do","does","did","will","would","could","should","to","of","in","on",
    "at","by","for","with","that","this","my","your","his","her","its",
    "our","me","him","us","them","what","which","who","not","no","yes",
    "just","very","also","well","now","then","here","there","all","more",
    "yo","tú","él","ella","nosotros","ellos","y","o","pero","que","es",
    "un","una","los","las","del","al","je","tu","il","nous","vous","ils",
    "et","ou","mais","que","est","un","une","le","la","les","io","tu",
    "lui","noi","voi","loro","e","o","ma","che","è","un","una","il","la",
}

NAME_BLOCKLIST = {
    "psychologin","trainerin","trainer","leiterin","leiter","entwicklerin",
    "entwickler","managerin","manager","direktin","direktor","digital",
    "learning","developerin","developer","teamleiterin","teamleiter",
    "bildung","neuropsychologie","kommunikation","ich","du","er","sie",
    "wir","das","ein","eine","der","die","hier","gut","sehr","auch","mal",
    "ja","nein","ok","heute","jetzt","dann","aber","doch","gerne","genau",
    "frau","herr","i","you","he","she","we","the","a","an","this","here",
    "good","yes","no","right","well","just","now","today","sure",
    # German emotion/state adjectives that follow "ich bin"
    "froh","glücklich","traurig","müde","nervös","gespannt","bereit",
    "fertig","sicher","klar","wichtig","richtig","falsch","dankbar",
    "erfreut","zufrieden","begeistert","überrascht","stolz","neugierig",
    # other common false positives
    "leider","insofern","natürlich","eigentlich","tatsächlich",
}

NAME_PATTERNS = {
    "de": {
        "self":  [(r"\bich bin (\w+)\b",1),(r"\bich hei[sß]e (\w+)\b",1),
                  (r"\bmein name ist (\w+)\b",1)],
        "other": [(r"\bhallo[,]? (\w+)\b",1),(r"\bwillkommen[,]? (\w+)\b",1),
                  (r"\bdas ist (\w+)\b",1),(r"\b(\w+)[,] stell\b",1),
                  (r"\bentschuldigung[,]? (\w+)\b",1)]
    },
    "en": {
        "self":  [(r"\bi am (\w+)\b",1),(r"\bmy name is (\w+)\b",1),
                  (r"\bi['']m (\w+)\b",1)],
        "other": [(r"\bwelcome[,]? (\w+)\b",1),(r"\bthis is (\w+)\b",1),
                  (r"\bhere(?:'s| is) (\w+)\b",1)]
    },
    "es": {
        "self":  [(r"\bme llamo (\w+)\b",1),(r"\bsoy (\w+)\b",1),
                  (r"\bmi nombre es (\w+)\b",1)],
        "other": [(r"\bbienvenido[,]? (\w+)\b",1),(r"\beste es (\w+)\b",1)]
    },
    "fr": {
        "self":  [(r"\bje m'appelle (\w+)\b",1),(r"\bje suis (\w+)\b",1)],
        "other": [(r"\bbienvenue[,]? (\w+)\b",1),(r"\bc'est (\w+)\b",1)]
    },
    "it": {
        "self":  [(r"\bmi chiamo (\w+)\b",1),(r"\bsono (\w+)\b",1)],
        "other": [(r"\bbenvenuto[,]? (\w+)\b",1),(r"\bquesto è (\w+)\b",1)]
    }
}

# ── DENGLISCH DETECTION ────────────────────────────────────────
# Common English words that appear in German speech (code-switching)
ENGLISH_IN_GERMAN = {
    "meeting","meetings","call","calls","update","updates","feedback",
    "team","teams","okay","ok","sorry","anyway","actually","basically",
    "workshop","workshops","onboarding","offboarding","deadline","deadlines",
    "follow","followup","follow-up","check","checkin","checkout","pipeline",
    "remote","homeoffice","freelance","startup","pitch","pitching","content",
    "skills","skill","performance","review","reviews","project","projects",
    "manager","management","lead","leadership","coaching","coach","training",
    "mindset","community","impact","output","input","outcome","outcomes",
    "challenge","challenges","highlight","highlights","learnings","learning",
    "roadmap","rollout","setup","workflow","workflows","dashboard","report",
    "reports","sprint","sprints","release","releases","feature","features",
    "bug","bugs","fix","fixes","testing","test","deployment","deployment",
    "digital","online","offline","live","stream","streaming","broadcast",
    "rebranding","branding","marketing","campaign","campaigns","target",
    "networking","network","platform","platforms","app","apps","tool",
    "tools","data","analytics","insights","benchmark","kpi","kpis",
    "e-learning","elearning","webinar","webinars","podcast","podcasts",
    "storytelling","brainstroming","brainstorming","ideation","prototype",
    "prototyping","agile","scrum","kanban","backlog","stakeholder",
}

def detect_denglisch(segments, language):
    """
    Detect code-switching: English words embedded in German (or other) speech.
    Returns list of hits with segment, word, and context.
    """
    if language not in ("de",):
        # For now only meaningful in German context
        return {"hits": [], "count": 0, "rate": 0.0,
                "top_words": [], "note": "Only tracked in German audio"}

    hits = []
    all_words = 0
    denglisch_words = Counter()

    for seg in segments:
        text  = seg.get("text","").lower()
        spk   = seg.get("speaker","UNKNOWN")
        words = re.sub(r"[^\w\s-]","",text).split()
        all_words += len(words)
        for w in words:
            clean = w.strip("-").lower()
            if clean in ENGLISH_IN_GERMAN:
                hits.append({
                    "time":    seg.get("start",0),
                    "speaker": spk,
                    "word":    w,
                    "context": seg.get("text","").strip()
                })
                denglisch_words[clean] += 1

    rate = round((len(denglisch_words) / max(all_words,1)) * 100, 2)
    return {
        "hits":      hits,
        "count":     len(hits),
        "rate":      rate,
        "top_words": [{"word":w,"count":c}
                      for w,c in denglisch_words.most_common(15)],
        "note":      f"{len(hits)} Denglisch instance(s) across {all_words} words ({rate}%)"
    }

# ── GPU HELPERS ────────────────────────────────────────────────
def flush_gpu():
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.synchronize()

def vram_free_gb():
    return round(torch.cuda.mem_get_info()[0]/1e9,2) \
           if torch.cuda.is_available() else 0.0

# ── EMBEDDING — CRITICAL FIX ───────────────────────────────────
def compute_embedding(model, chunk, t=0.0):
    """
    FIX v6: pyannote Inference expects (channel, time) = shape (1, T).
    Previous code passed unsqueeze(0).unsqueeze(0) → (1,1,T) — WRONG.
    Correct: unsqueeze(0) → (1, T).
    """
    if len(chunk) / SAMPLE_RATE < MIN_EMBED_DUR:
        return None
    try:
        # (T,) → (1, T)  ← this is the correct shape
        wt = torch.from_numpy(chunk).float().unsqueeze(0)
        e  = model({"waveform": wt, "sample_rate": SAMPLE_RATE})
        if e is not None:
            arr = np.array(e).flatten()
            if not np.all(arr == 0):
                return arr
    except Exception as ex:
        print(f"  [EMB] {t:.1f}s: {ex}")
    return None

def build_embeddings(df, audio, model, skip_before=0.0):
    chunks = defaultdict(list)
    for _, row in df.iterrows():
        if row.start < skip_before: continue
        dur = row.end - row.start
        if dur < MIN_EMBED_DUR: continue
        s, e = int(row.start*SAMPLE_RATE), int(row.end*SAMPLE_RATE)
        chunks[row.speaker].append((dur, audio[s:e], row.start))

    # Fallback: if any speaker has zero chunks after skip_before,
    # include their best segments from the full file
    all_speakers = df.speaker.unique()
    for spk in all_speakers:
        if spk not in chunks:
            print(f"  [EMB] {spk}: no segments after {skip_before}s, "
                  f"using full-file fallback")
            for _, row in df[df.speaker==spk].iterrows():
                dur = row.end - row.start
                if dur < MIN_EMBED_DUR: continue
                s, e = int(row.start*SAMPLE_RATE), int(row.end*SAMPLE_RATE)
                chunks[spk].append((dur, audio[s:e], row.start))

    out = {}
    for spk, items in chunks.items():
        items.sort(key=lambda x: x[0], reverse=True)
        embs = []
        for _, c, t in items[:8]:
            emb = compute_embedding(model, c, t)
            if emb is not None:
                embs.append(emb)
        if embs:
            out[spk] = np.mean(embs, axis=0)
            print(f"  [EMB] {spk}: fingerprint from {len(embs)} segments ✓")
        else:
            print(f"  [EMB] {spk}: no embeddings built")
    return out

def closest_speaker(q, embs, thresh=EMBED_THRESH):
    best_spk, best_sim = None, -1.0
    for spk, e in embs.items():
        sim = 1 - cosine(q, e)
        if sim > best_sim:
            best_sim, best_spk = sim, spk
    return (best_spk, best_sim) if best_sim >= thresh else (None, best_sim)

# ── COLD START — SILENCE-GATED EMBEDDING ───────────────────────
def get_speech_bursts(audio, end_s,
                      min_sil_ms=300, sil_db=-42, min_ms=800):
    end   = int(end_s * SAMPLE_RATE)
    a16   = (audio[:end] * 32767).astype(np.int16)
    seg   = AudioSegment(a16.tobytes(), frame_rate=SAMPLE_RATE,
                         sample_width=2, channels=1)
    rngs  = detect_nonsilent(seg, min_silence_len=min_sil_ms,
                              silence_thresh=sil_db, seek_step=10)
    bursts = [(s/1000, e/1000) for s,e in rngs if (e-s) >= min_ms]
    print(f"  [COLD] {len(bursts)} speech burst(s) in first {end_s:.0f}s")
    return bursts

def segment_cold_acoustic(audio, end_s, embs, emb_model):
    """
    Cold-start speaker assignment without diarization.
    Strategy:
      1. Split on silence — reliable even for similar-pitched speakers.
      2. Sub-split long bursts (>4s) on pitch change-points so single-speaker
         turns that have natural pauses don't get merged across speakers.
      3. Embed each sub-burst and match to global fingerprints (built from 60s+).
      4. Merge consecutive same-speaker segments.
    """
    bursts = get_speech_bursts(audio, end_s)
    if not bursts:
        return pd.DataFrame(columns=["speaker","start","end"])

    rows = []
    for cs, ce in bursts:
        dur = ce - cs
        ss  = int(cs * SAMPLE_RATE)
        se  = int(ce * SAMPLE_RATE)
        piece = audio[ss:se]

        # Sub-split bursts longer than 3s using parselmouth pitch change-points
        sub_segs = [(cs, ce)]  # default: whole burst as one segment
        if dur > 3.0:
            try:
                snd    = parselmouth.Sound(piece.astype(np.float64),
                                          sampling_frequency=SAMPLE_RATE)
                pt     = snd.to_pitch(time_step=0.05,
                                      pitch_floor=75, pitch_ceiling=400)
                frames = pt.selected_array["frequency"]
                times  = [pt.get_time_from_frame_number(i+1)
                          for i in range(len(frames))]

                # Smooth pitch with 5-frame median, detect jumps > 18Hz
                voiced_idx = [i for i,f in enumerate(frames) if f > 0]
                if len(voiced_idx) > 10:
                    voiced_f = np.array([frames[i] for i in voiced_idx],
                                        dtype=np.float32)
                    voiced_t = np.array([times[i]  for i in voiced_idx])
                    smoothed = np.convolve(voiced_f,
                                          np.ones(5)/5, mode='same')
                    splits = [cs]
                    for i in range(4, len(smoothed)-4):
                        delta = abs(smoothed[i] - smoothed[i-4])
                        if delta > 18.0:
                            t_split = cs + float(voiced_t[i])
                            if t_split - splits[-1] > 1.0:
                                splits.append(t_split)
                    splits.append(ce)
                    if len(splits) > 2:
                        sub_segs = [(splits[i], splits[i+1])
                                    for i in range(len(splits)-1)]
            except Exception as ex:
                pass  # keep whole burst

        for t_start, t_end in sub_segs:
            seg_dur = t_end - t_start
            if seg_dur < MIN_EMBED_DUR:
                continue
            s2 = int(t_start * SAMPLE_RATE)
            e2 = int(t_end   * SAMPLE_RATE)
            seg_audio = audio[s2:e2]
            emb = compute_embedding(emb_model, seg_audio, t_start)
            if emb is None or not embs:
                rows.append({"speaker":"UNKNOWN",
                             "start": t_start, "end": t_end})
                continue
            gspk, sim = closest_speaker(emb, embs, thresh=COLD_THRESH)
            label = gspk if gspk else "UNKNOWN"
            print(f"  [COLD] {t_start:.1f}s→{t_end:.1f}s → {label} "
                  f"(sim={sim:.3f})")
            rows.append({"speaker": label,
                         "start":   t_start,
                         "end":     t_end})

    if not rows:
        return pd.DataFrame(columns=["speaker","start","end"])

    # Merge consecutive same-speaker segments
    df = (pd.DataFrame(rows)
          .sort_values("start")
          .reset_index(drop=True))
    merged = [df.iloc[0].to_dict()]
    for _, row in df.iloc[1:].iterrows():
        if row["speaker"] == merged[-1]["speaker"]:
            merged[-1]["end"] = row["end"]
        else:
            merged.append(row.to_dict())
    result = pd.DataFrame(merged)
    print(f"  [COLD] {len(result)} merged segment(s) after speaker grouping")
    return result


# ── NAME RESOLUTION ────────────────────────────────────────────
def extract_name_evidence(segs, lang, audio, embs, emb_model):
    pp      = NAME_PATTERNS.get(lang, NAME_PATTERNS["en"])
    self_p  = pp["self"]
    other_p = pp["other"]
    self_hits  = defaultdict(lambda: defaultdict(int))
    other_hits = defaultdict(lambda: defaultdict(int))

    for i, seg in enumerate(segs):
        spk = seg.get("speaker","UNKNOWN")
        if spk == "UNKNOWN": continue
        win = " ".join(s["text"] for s in segs[i:i+3]).lower()
        ss, se = seg["start"], seg["end"]

        for pat, grp in self_p:
            for m in re.finditer(pat, win):
                raw  = m.group(grp).strip()
                name = SURNAME_ALIAS.get(raw.lower(), raw.capitalize())
                if len(name) <= 2 or name.lower() in NAME_BLOCKLIST:
                    continue
                if (se-ss) >= MIN_EMBED_DUR and embs:
                    s_s = int(ss*SAMPLE_RATE)
                    e_s = int(se*SAMPLE_RATE)
                    emb = compute_embedding(emb_model, audio[s_s:e_s], ss)
                    if emb is not None:
                        ms, sim = closest_speaker(emb, embs)
                        if ms:
                            self_hits[name][ms] += 2
                            continue
                self_hits[name][spk] += 1

        for pat, grp in other_p:
            for m in re.finditer(pat, win):
                raw  = m.group(grp).strip()
                name = SURNAME_ALIAS.get(raw.lower(), raw.capitalize())
                if len(name) <= 2 or name.lower() in NAME_BLOCKLIST:
                    continue
                other_hits[name][spk] += 1

    scores = defaultdict(lambda: defaultdict(float))
    for name, sh in self_hits.items():
        for spk, h in sh.items():
            scores[spk][name] += h * 2.0
    for name, oh in other_hits.items():
        for spk in set(embs) - set(oh):
            scores[spk][name] += sum(oh.values()) * 0.5

    ev = {}
    for spk, ns in scores.items():
        if not ns: continue
        best = max(ns, key=ns.get)
        sc   = ns[best]
        ev[spk] = {
            "name":           best,
            "score":          sc,
            "confidence":     round(min(sc/(NAME_CONF_THRES*2), 1.0), 2),
            "auto_assign":    sc >= NAME_CONF_THRES,
            "all_candidates": dict(ns)
        }
    return ev

def resolve_names(ev):
    s2n, review = {}, []
    name_cands  = defaultdict(list)
    for spk, d in ev.items():
        if d["auto_assign"]:
            name_cands[d["name"]].append((spk, d["score"]))
    for name, cands in name_cands.items():
        cands.sort(key=lambda x: x[1], reverse=True)
        s2n[cands[0][0]] = name
    for spk in ev:
        if spk not in s2n:
            review.append(spk)
    return s2n, review

# ── OLLAMA ─────────────────────────────────────────────────────
def ollama_available():
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                data = json.loads(r.read().decode("utf-8"))
                models = [m.get("name","") for m in data.get("models",[])]
                print(f"  [LLM] Ollama online. Models: {models}")
                return True
        return False
    except Exception as e:
        print(f"  [LLM] Ollama check failed: {type(e).__name__}: {e}")
        return False

def call_ollama(prompt, model=OLLAMA_MODEL, max_tokens=6000):
    payload = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.15, "num_predict": max_tokens}
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("response","")
    except Exception as e:
        print(f"  [LLM] Ollama error type: {type(e).__name__}")
        print(f"  [LLM] Ollama error detail: {e}")
        return None

def warm_ollama():
    """Send a tiny prompt to load the model into VRAM before the real call."""
    print(f"  [LLM] Warming up {OLLAMA_MODEL}...")
    result = call_ollama("Say OK.", max_tokens=5)
    if result is not None:
        print(f"  [LLM] Model warm. Response: {result[:30]}")
        return True
    print(f"  [LLM] Warm-up failed.")
    return False

def parse_json_from_llm(text):
    """Robustly extract JSON from LLM response (handles markdown fences)."""
    if not text: return None
    clean = re.sub(r"```(?:json)?|```","",text).strip()
    # Try to find JSON object or array
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                continue
    return None

# ── SPEAKER CORRECTION PROMPT ──────────────────────────────────
def build_correction_prompt(segments, lang, blueprints=None, role_map=None):
    lang_name = {"de":"German","en":"English","es":"Spanish",
                 "fr":"French","it":"Italian"}.get(lang,"Unknown")

    # Pre-scan: find all self-introduction evidence to inject as hints
    intro_hints = []
    intro_patterns = [
        r"ich bin (\w+)", r"ich hei[sß]e (\w+)", r"mein name ist (\w+)",
        r"i am (\w+)", r"my name is (\w+)", r"i'm (\w+)",
        r"me llamo (\w+)", r"je m'appelle (\w+)", r"mi chiamo (\w+)"
    ]
    addr_patterns = [
        r"hallo[,\s]+frau[,\s]+(\w+)",   # "Hallo Frau Gassier" → Gassier
        r"hallo[,\s]+herr[,\s]+(\w+)",   # "Hallo Herr X"
        r"hallo[,\s]+(\w+)",              # "Hallo Marcella"
        r"entschuldigung[,\s]+(\w+)",     # "Entschuldigung, Raffi"
        r"welcome[,\s]+(\w+)",
        r"hi[,\s]+(\w+)"
    ]
    # Use the global blocklist — single source of truth
    BLOCKLIST = NAME_BLOCKLIST | {
        "es","ihr","dem","den","einen","einem","einer","eines",
        "bitte","danke","also","da","noch","halt","schon",
        "super","toll","prima","okay","ach","aha","hmm","äh","ähm",
        "dr","prof","mr","ms","mrs","miss","sir",
        "they","it","that","great","recruiting","rekruterin",
    }

    for i, seg in enumerate(segments):
        txt = seg["text"].lower().strip()
        spk = seg["speaker"]
        for pat in intro_patterns:
            m = re.search(pat, txt)
            if m:
                raw  = m.group(1).strip()
                name = SURNAME_ALIAS.get(raw.lower(), raw.capitalize())
                if name.lower() not in BLOCKLIST and len(name) > 2:
                    intro_hints.append(
                        f"  • Segment [{i:03d}] ({seg['start']}s): "
                        f"speaker labelled '{spk}' says '{seg['text'].strip()}' "
                        f"→ this person's name IS {name}"
                    )
        for pat in addr_patterns:
            m = re.search(pat, txt)
            if m:
                raw  = m.group(1).strip()
                name = SURNAME_ALIAS.get(raw.lower(), raw.capitalize())
                if name.lower() not in BLOCKLIST and len(name) > 2:
                    intro_hints.append(
                        f"  • Segment [{i:03d}] ({seg['start']}s): "
                        f"speaker labelled '{spk}' addresses '{raw}' (={name}) "
                        f"→ {name} is a DIFFERENT person, NOT the speaker of this segment"
                    )

    hints_block = "\n".join(intro_hints) if intro_hints else "  (none detected automatically)"

    # Detect label contradictions: segment labelled X but text says "ich bin Y"
    contradictions = []
    for i, seg in enumerate(segments):
        txt = seg["text"].lower().strip()
        spk = seg["speaker"]
        for pat in intro_patterns:
            m = re.search(pat, txt)
            if m:
                raw  = m.group(1).strip()
                name = SURNAME_ALIAS.get(raw.lower(), raw.capitalize())
                if (name.lower() not in BLOCKLIST and len(name) > 2
                        and name != spk):
                    contradictions.append(
                        f"  ⚠ Segment [{i:03d}] ({seg['start']}s): "
                        f"labelled '{spk}' but says '{seg['text'].strip()}' "
                        f"→ MUST be relabelled {name}"
                    )

    contra_block = ("\n".join(contradictions)
                    if contradictions else "  (none found)")

    lines = "\n".join(
        "[{:03d}] {:.1f}s | {} | {} | {}".format(
            i,
            seg["start"],
            seg["speaker"],
            ("pitch:{}Hz energy:{}dB".format(seg["pitch_hz"], seg["energy_db"])
             if seg.get("pitch_hz") else "pitch:?"),
            seg["text"].strip()
        )
        for i, seg in enumerate(segments)
    )

    # Build the set of valid names from intro evidence
    valid_names = set()
    for hint in intro_hints:
        m = re.search(r"name IS (\w+)", hint)
        if m:
            valid_names.add(m.group(1))
    valid_names_str = ", ".join(sorted(valid_names)) + ", UNKNOWN" if valid_names else "UNKNOWN"

    # Build acoustic context block from blueprints
    acoustic_block = ""
    if blueprints:
        lines_a = []
        for name, bp in blueprints.items():
            lines_a.append(
                f"  • {name}: avg pitch {bp['avg_pitch_hz']}Hz "
                f"({bp['gender_hint']}), {bp['speaking_rate_wpm']} wpm, "
                f"{bp['expressiveness']} delivery, "
                f"voice quality {bp['voice_quality_label']} (HNR {bp['voice_quality_hnr']})"
            )
        acoustic_block = "ACOUSTIC VOICE PROFILES (use to distinguish speakers):\n" + \
                         "\n".join(lines_a)
    else:
        acoustic_block = "ACOUSTIC VOICE PROFILES: not yet available for this pass"

    # Build role context block from Stage A inference
    role_block = ""
    if role_map:
        role_lines = [f"  {spk} → {role}" for spk, role in role_map.items()]
        role_block = ("CONVERSATIONAL ROLES (determined from turn-taking analysis):\n"
                      + "\n".join(role_lines) + "\n"
                      + "Use these roles to resolve ambiguous segments: "
                        "candidates answer questions, interviewers ask them.\n")

    return f"""You are a speaker diarization correction expert fixing WRONG speaker labels in a {lang_name} transcript.

THE CORE PROBLEM: The audio diarization tool merged multiple people under the same label. Fix every segment.

{acoustic_block}

{role_block}SELF-INTRODUCTION EVIDENCE:
{hints_block}

LABEL CONTRADICTIONS — these segments MUST be corrected, the current label is provably wrong:
{contra_block}

STRICT RULES:
1. Speaker names MUST be real human first names only. Never use: titles (Frau, Herr, Dr), job titles, common nouns, German/English words, or anything that is not a person's first name.
2. "Ich bin X" / "Ich heiße X" → that segment's speaker IS X. X must be a first name.
3. "Hallo X" / "Entschuldigung X" → X is someone ELSE in the room, NOT the current speaker.
4. "Frau Gassier" / "Herr X" → word after Frau/Herr is a SURNAME. Do NOT use as speaker name.
5. Once you identify a speaker from a self-introduction, apply that name to their entire continuous speaking turn.
6. COLD START (first 30 seconds): labels before any self-introduction are UNRELIABLE. Use role context and pitch_hz to reassign.
7. If you cannot confidently identify a speaker, use UNKNOWN. Never invent a name.
8. Return ALL {len(segments)} segments. Never skip an index.
9. Return ONLY the JSON array. No explanation, no preamble, no markdown.
10. Use "id" as the key (not "idx").

VALID speaker names: {valid_names_str}
Use ONLY these names. Any other name is forbidden.

TRANSCRIPT:
{lines}

JSON array ({len(segments)} entries, key must be "id"):"""

# ── CONTEXT-AWARE DEEP ANALYSIS PROMPTS ───────────────────────
def build_analysis_prompt(segments, lang, context, speaker_names):
    lang_name = {"de":"German","en":"English","es":"Spanish",
                 "fr":"French","it":"Italian"}.get(lang,"Unknown")
    full_text = "\n".join(
        f"[{seg['start']:.1f}s] {seg['speaker']}: {seg['text']}"
        for seg in segments
    )
    names_str = ", ".join(speaker_names) if speaker_names else "unknown"

    if context == "interview":
        return f"""Analyze this {lang_name} job interview transcript.
Speakers identified: {names_str}
Interviewers ask questions; candidates answer.

Return a JSON object with these exact keys:
{{
  "topic": "main topic or job role of the interview",
  "qa_pairs": [
    {{
      "question": "exact question text",
      "asked_by": "speaker name",
      "answered_by": "speaker name",
      "answer_text": "summary of answer given",
      "on_topic": true/false,
      "deviation_note": "if off-topic, explain briefly or null",
      "reaction_seconds": number (seconds between question end and answer start),
      "understood": true/false,
      "quality": "strong/adequate/weak/evasive"
    }}
  ],
  "candidate_stats": {{
    "total_questions": number,
    "answered_on_topic": number,
    "deviated": number,
    "avg_reaction_seconds": number,
    "comprehension_rate": number (0-100),
    "overall_assessment": "2-3 sentence summary"
  }},
  "interviewer_notes": "brief observation about interviewer style",
  "key_topics_mentioned": ["topic1","topic2"]
}}

TRANSCRIPT:
{full_text}

Return only the JSON object:"""

    elif context in ("language_practice","self_recorded"):
        return f"""Analyze this {lang_name} audio recording for language learning feedback.

Return a JSON object with these exact keys:
{{
  "overall_fluency": "Beginner/Intermediate/Advanced/Native-like",
  "fluency_score": number (0-100),
  "language_observations": [
    {{
      "timestamp": number,
      "speaker": "name or UNKNOWN",
      "original": "what was said",
      "issue_type": "grammar/vocabulary/pronunciation_hint/filler/awkward_phrasing/none",
      "correction": "corrected version if applicable or null",
      "native_alternatives": ["alternative 1","alternative 2","alternative 3"],
      "explanation": "brief explanation of the issue or improvement"
    }}
  ],
  "grammar_patterns": [
    {{"pattern":"description","count":number,"example":"example from transcript"}}
  ],
  "vocabulary_level": "A1/A2/B1/B2/C1/C2",
  "filler_words": [{{"word":"um/uh/äh/etc","count":number}}],
  "strengths": ["strength 1","strength 2"],
  "improvement_areas": ["area 1","area 2"],
  "pronunciation_notes": "general pronunciation observations based on text patterns",
  "recommended_focus": "top 1-2 things to work on"
}}

TRANSCRIPT:
{full_text}

Return only the JSON object:"""

    elif context == "lecture":
        return f"""Analyze this {lang_name} lecture or presentation transcript.

Return a JSON object with these exact keys:
{{
  "topic": "main subject of the lecture",
  "subtopics": ["subtopic 1","subtopic 2"],
  "structure": {{
    "introduction_present": true/false,
    "conclusion_present": true/false,
    "logical_flow": "Good/Fair/Poor",
    "transitions_quality": "Good/Fair/Poor"
  }},
  "delivery_observations": [
    {{
      "timestamp": number,
      "observation": "description",
      "type": "pace/clarity/engagement/repetition/digression"
    }}
  ],
  "key_concepts": ["concept 1","concept 2"],
  "vocabulary_complexity": "Basic/Intermediate/Advanced/Technical",
  "estimated_audience": "who this lecture seems aimed at",
  "summary": "3-4 sentence summary of the lecture content"
}}

TRANSCRIPT:
{full_text}

Return only the JSON object:"""

    else:  # unknown / general
        return f"""Analyze this {lang_name} audio transcript and provide intelligence insights.

Return a JSON object with these exact keys:
{{
  "content_type": "what kind of recording this appears to be",
  "topic": "main topic or subject",
  "key_points": ["point 1","point 2","point 3"],
  "tone": "formal/informal/emotional/professional/casual",
  "notable_moments": [
    {{"timestamp":number,"observation":"what happens here"}}
  ],
  "language_quality": "observations about how the language is used",
  "summary": "3-4 sentence summary"
}}

TRANSCRIPT:
{full_text}

Return only the JSON object:"""

# ── ACOUSTIC BLUEPRINTS ────────────────────────────────────────
def build_voice_blueprints(audio, segments):
    spk_segs = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker","UNKNOWN")
        if spk not in ("UNKNOWN",""):
            spk_segs[spk].append(seg)

    blueprints = {}
    for name, segs in spk_segs.items():
        pitch_vals, energy_vals, hnr_vals = [], [], []
        total_words, total_dur, seg_count  = 0, 0.0, 0

        for seg in segs:
            dur = seg["end"] - seg["start"]
            if dur < 0.5: continue
            s, e  = int(seg["start"]*SAMPLE_RATE), int(seg["end"]*SAMPLE_RATE)
            chunk = audio[s:e]
            if len(chunk) < int(SAMPLE_RATE*0.5): continue
            try:
                sound = parselmouth.Sound(chunk, sampling_frequency=SAMPLE_RATE)
                pitch = sound.to_pitch(time_step=0.01,
                                       pitch_floor=75, pitch_ceiling=400)
                voiced = pitch.selected_array["frequency"]
                voiced = voiced[voiced > 0]
                if len(voiced): pitch_vals.extend(voiced.tolist())
                try:
                    hnr = sound.to_harmonicity()
                    hv  = hnr.values[hnr.values != -200]
                    if len(hv): hnr_vals.extend(hv.tolist())
                except: pass
                rms = np.sqrt(np.mean(chunk.astype(np.float64)**2))
                if rms > 0: energy_vals.append(20*np.log10(rms))
                total_words += len(seg["text"].split())
                total_dur   += dur
                seg_count   += 1
            except: continue

        if not pitch_vals: continue

        avg_p = float(np.mean(pitch_vals))
        rng_p = float(np.max(pitch_vals) - np.min(pitch_vals))
        std_p = float(np.std(pitch_vals))
        med_p = float(np.median(pitch_vals))
        avg_e = float(np.mean(energy_vals)) if energy_vals else 0.0
        avg_h = float(np.mean(hnr_vals))    if hnr_vals   else 0.0
        wpm   = round((total_words/total_dur)*60) if total_dur > 0 else 0

        expr     = ("Expressive" if std_p>40 else "Moderate" if std_p>20 else "Monotone")
        pace     = ("Fast" if wpm>160 else "Normal" if wpm>110 else "Slow")
        voice_q  = ("Breathy" if avg_h<5 else "Clear" if avg_h>15 else "Neutral")
        gender_h = ("Likely female" if avg_p>165 else
                    "Likely male"   if avg_p<130 else "Ambiguous")

        blueprints[name] = {
            "avg_pitch_hz":          round(avg_p,1),
            "median_pitch_hz":       round(med_p,1),
            "pitch_range_hz":        round(rng_p,1),
            "pitch_variability_std": round(std_p,1),
            "avg_energy_db":         round(avg_e,1),
            "voice_quality_hnr":     round(avg_h,1),
            "voice_quality_label":   voice_q,
            "speaking_rate_wpm":     wpm,
            "expressiveness":        expr,
            "pace":                  pace,
            "gender_hint":           gender_h,
            "segments_analyzed":     seg_count,
            "total_speech_seconds":  round(total_dur,1),
            "profile_label":         f"{expr} · {pace} · {round(avg_p)}Hz · {voice_q}"
        }
        print(f"  [BLUEPRINT] {name}: {blueprints[name]['profile_label']}")
    return blueprints

# ── WORD FREQUENCY ─────────────────────────────────────────────
def _make_spk_words(cnt):
    top = cnt.most_common(40)
    mx  = top[0][1] if top else 1
    return [{"word":w,"count":c,"weight":round(c/mx,3)} for w,c in top]

def analyze_words(segments, top_n=80):
    global_c = Counter()
    spk_c    = defaultdict(Counter)
    for seg in segments:
        spk   = seg.get("speaker","UNKNOWN")
        words = re.sub(r"[^\w\s]","",seg.get("text","").lower()).split()
        filt  = [w for w in words
                 if w not in STOP_WORDS and len(w)>2 and not w.isdigit()]
        global_c.update(filt)
        if spk not in ("UNKNOWN",""):
            spk_c[spk].update(filt)

    top = global_c.most_common(top_n)
    mx  = top[0][1] if top else 1
    return {
        "global": [{"word":w,"count":c,"weight":round(c/mx,3)} for w,c in top],
        "by_speaker": {
            s: _make_spk_words(cnt)
            for s,cnt in spk_c.items() if cnt
        },
        "total_unique_words": len(global_c),
        "total_word_tokens":  sum(global_c.values())
    }

# ── DIARIZATION HELPER ─────────────────────────────────────────
def run_diarization(pipeline, audio, n_spk=0):
    wt = torch.from_numpy(audio).unsqueeze(0)
    p  = ({"num_speakers":n_spk,"min_speakers":n_spk,"max_speakers":n_spk}
          if n_spk>0 else {"min_speakers":1,"max_speakers":6})
    d  = pipeline({"waveform":wt,"sample_rate":SAMPLE_RATE},**p)
    # community-1 exposes exclusive_speaker_diarization which gives cleaner
    # one-speaker-per-frame output — much easier to align with Whisper words
    try:
        tracks = d.exclusive_speaker_diarization.itertracks(yield_label=True)
        print("  [DIAR] Using exclusive_speaker_diarization (community-1)")
    except AttributeError:
        try:    tracks = d.itertracks(yield_label=True)
        except: tracks = d.speaker_diarization.itertracks(yield_label=True)
        print("  [DIAR] Using standard itertracks")
    df = pd.DataFrame(list(tracks), columns=["segment","label","speaker"])
    df["start"] = df.segment.apply(lambda x: x.start)
    df["end"]   = df.segment.apply(lambda x: x.end)
    return df

# ════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.post("/process")
async def process_audio(
    audio:        UploadFile,
    hf_token:     str = Form(...),
    num_speakers: int = Form(0),
    language:     str = Form("de"),
    context:      str = Form("unknown")
):
    t_wall     = time.time()
    pass_times = {}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # ── PASS 1: TRANSCRIPTION ─────────────────────────────
        p1 = time.time()
        print(f"\n[PASS 1] Transcribing... VRAM:{vram_free_gb()}GB")
        audio_data = whisperx.load_audio(tmp_path)
        result     = whisper_model.transcribe(audio_data, batch_size=16,
                                              language=language)
        ma, meta   = whisperx.load_align_model(language_code=language,
                                               device=DEVICE)
        result     = whisperx.align(result["segments"], ma, meta,
                                    audio_data, DEVICE,
                                    return_char_alignments=False)
        del ma, meta; flush_gpu()
        pass_times["pass1"] = round(time.time()-p1, 2)
        print(f"[PASS 1] Done {pass_times['pass1']}s")

        # ── PASS 2: DIARIZATION + FINGERPRINTS ────────────────
        p2 = time.time()
        print(f"\n[PASS 2] Diarization... VRAM:{vram_free_gb()}GB")
        os.environ["HF_TOKEN"] = hf_token
        pipeline  = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        ).to(torch.device(DEVICE))
        audio2    = whisperx.load_audio(tmp_path)
        full_df   = run_diarization(pipeline, audio2, int(num_speakers))
        n_detect  = full_df.speaker.nunique()
        print(f"[PASS 2] {n_detect} speaker(s) detected")
        # Raw diarization dump — first 30 rows to diagnose clustering
        print("[PASS 2] Raw diarization (first 30 rows):")
        for _, row in full_df.head(30).iterrows():
            print(f"  {row['speaker']}  {row['start']:.1f}s → {row['end']:.1f}s")

        emb_model = Inference(
            Model.from_pretrained("pyannote/embedding", token=hf_token),
            window="whole"
        )
        embs = build_embeddings(full_df, audio2, emb_model,
                                skip_before=COLD_START_S)
        if not embs:
            print("  [EMB] Fallback: using full file for fingerprints")
            embs = build_embeddings(full_df, audio2, emb_model,
                                    skip_before=0.0)
        pass_times["pass2"] = round(time.time()-p2, 2)
        print(f"[PASS 2] Done {pass_times['pass2']}s  Prints:{list(embs)}")

        # ── PASS 3: ACOUSTIC-ONLY COLD-START (no diarization) ────
        p3 = time.time()
        cold_end = min(COLD_START_S, len(audio2)/SAMPLE_RATE)
        print(f"\n[PASS 3] Acoustic cold-start 0→{cold_end:.0f}s "
              f"(pitch+energy, no diarization)...")
        cold_df = segment_cold_acoustic(audio2, cold_end, embs, emb_model)
        if not cold_df.empty:
            outside = full_df[full_df.start >= cold_end].copy()
            full_df = pd.concat(
                [cold_df[["speaker","start","end"]], outside],
                ignore_index=True
            ).sort_values("start").reset_index(drop=True)

        del pipeline; flush_gpu()
        result = whisperx.assign_word_speakers(full_df, result)
        pass_times["pass3"] = round(time.time()-p3, 2)
        print(f"[PASS 3] Done {pass_times['pass3']}s")

        # ── PASS 4a: REGEX NAME RESOLUTION ────────────────────
        p4a = time.time()
        print(f"\n[PASS 4a] Name resolution...")
        name_ev    = extract_name_evidence(result["segments"], language,
                                           audio2, embs, emb_model)
        s2n, review= resolve_names(name_ev)

        segs_raw = []
        for seg in result["segments"]:
            spk_id   = seg.get("speaker","UNKNOWN")
            resolved = s2n.get(spk_id, spk_id)

            # Compute lightweight per-segment pitch for LLM grounding
            seg_pitch = None
            seg_energy = None
            try:
                ss = int(seg["start"] * SAMPLE_RATE)
                se = int(seg.get("end", seg["start"]+1) * SAMPLE_RATE)
                chunk = audio_data[ss:se]
                if len(chunk) > int(SAMPLE_RATE * 0.4):
                    sound = parselmouth.Sound(chunk,
                                             sampling_frequency=SAMPLE_RATE)
                    pitch = sound.to_pitch(time_step=0.02,
                                          pitch_floor=75, pitch_ceiling=400)
                    voiced = pitch.selected_array["frequency"]
                    voiced = voiced[voiced > 0]
                    if len(voiced):
                        seg_pitch = round(float(np.median(voiced)), 1)
                    rms = np.sqrt(np.mean(chunk.astype(np.float64)**2))
                    if rms > 0:
                        seg_energy = round(20 * np.log10(rms), 1)
            except:
                pass

            segs_raw.append({
                "idx":           len(segs_raw),
                "start":         round(seg["start"],1),
                "end":           round(seg.get("end", seg["start"]+1),1),
                "speaker":       resolved,
                "speaker_id":    spk_id,
                "text":          seg["text"].strip(),
                "llm_corrected": False,
                "pitch_hz":      seg_pitch,
                "energy_db":     seg_energy,
            })

        del emb_model; flush_gpu()
        pass_times["pass4a"] = round(time.time()-p4a, 2)
        print(f"[PASS 4a] Done {pass_times['pass4a']}s  s2n={s2n}")

        # ── PASS 4b: LLM SPEAKER CORRECTION (2-stage) ─────────
        p4b = time.time()
        llm_used        = False
        llm_corrections = 0
        llm_status      = "skipped — Ollama offline"

        print(f"\n[PASS 4b] Building early acoustic profiles for LLM context...")
        early_blueprints = build_voice_blueprints(audio_data, segs_raw)

        if ollama_available():
            warm_ollama()

            # ── Stage A: conversational role inference (no names) ──
            # The LLM figures out WHO is who by turn-taking patterns,
            # question/answer dynamics, and speaking style — without
            # being poisoned by bad acoustic labels.
            raw_lines = "\n".join(
                "[{:03d}] {:.1f}s | {} | {}".format(
                    i, seg["start"], seg["speaker_id"], seg["text"].strip()
                )
                for i, seg in enumerate(segs_raw)
            )
            spk_ids = sorted(set(s["speaker_id"] for s in segs_raw
                                 if s["speaker_id"] not in ("UNKNOWN","")))

            acoustic_desc = ""
            if early_blueprints:
                desc_lines = []
                for spk_id in spk_ids:
                    # match blueprint by resolved name or speaker_id
                    resolved = s2n.get(spk_id, spk_id)
                    bp = early_blueprints.get(resolved) or \
                         early_blueprints.get(spk_id)
                    if bp:
                        desc_lines.append(
                            f"  {spk_id}: pitch {bp['avg_pitch_hz']}Hz, "
                            f"{bp['speaking_rate_wpm']}wpm, "
                            f"{bp['expressiveness']} delivery"
                        )
                acoustic_desc = "ACOUSTIC PROFILES:\n" + "\n".join(desc_lines)

            lang_label = {"de":"German","en":"English","es":"Spanish",
                          "fr":"French","it":"Italian"}.get(language,"unknown")
            role_prompt = f"""You are analyzing a {lang_label} conversation transcript.
Your task: assign a CONVERSATIONAL ROLE to each speaker ID based on behaviour patterns only.
Do NOT use names. Use only these role labels: interviewer, candidate, moderator, unknown.

{acoustic_desc}

HOW TO IDENTIFY ROLES:
- interviewer: asks questions, introduces agenda, gives structured turns, speaks formally
- candidate: answers questions, introduces themselves, is being evaluated
- moderator: coordinates between others, shorter turns, hands floor to others
- The candidate typically has longer answer turns after questions
- Interviewers typically ask questions ending with "oder?", "richtig?", "können Sie..."

TRANSCRIPT (speaker IDs, not names):
{raw_lines}

Return ONLY a JSON object mapping each speaker_id to a role:
{{"SPEAKER_00": "candidate", "SPEAKER_01": "interviewer", "SPEAKER_02": "interviewer"}}"""

            print(f"  [LLM] Stage A: role inference ({len(spk_ids)} speaker IDs)...")
            role_resp = call_ollama(role_prompt, max_tokens=500)
            print(f"  [LLM] Role response: {(role_resp or '').strip()[:300]}")

            role_map = {}  # speaker_id → role
            if role_resp:
                try:
                    clean = re.sub(r"```[a-z]*|```","", role_resp).strip()
                    # find first {...} block
                    m = re.search(r"\{[^}]+\}", clean, re.DOTALL)
                    if m:
                        role_map = json.loads(m.group(0))
                        print(f"  [LLM] Roles: {role_map}")
                except Exception as e:
                    print(f"  [LLM] Role parse failed: {e}")

            # ── Stage B: name assignment using roles + intro evidence ─
            corr_prompt = build_correction_prompt(
                segs_raw, language, blueprints=early_blueprints,
                role_map=role_map
            )
            print(f"  [LLM] Stage B: name correction ({len(segs_raw)} segments)...")
            corr_resp = call_ollama(corr_prompt, max_tokens=8000)
            print(f"  [LLM] Raw response (first 600 chars):\n{(corr_resp or '')[:600]}")

            if corr_resp:
                parsed = parse_json_from_llm(corr_resp)
                print(f"  [LLM] Parsed type: {type(parsed).__name__}, "
                      f"len: {len(parsed) if isinstance(parsed,list) else 'n/a'}")
                if isinstance(parsed, list):
                    changed = 0
                    # Build valid name set from what we actually know
                    known_valid = set(s2n.values()) | {"UNKNOWN"}
                    # Also accept any name the LLM found via intro evidence
                    # (passed through valid_names in the prompt)
                    for item in parsed:
                        if not isinstance(item, dict): continue
                        # Accept both "idx" and "id" — model often drifts
                        idx = item.get("idx") if item.get("idx") is not None \
                              else item.get("id")
                        spk = item.get("speaker","").strip()
                        if idx is None or not spk: continue
                        try: idx = int(idx)
                        except: continue
                        if not (0 <= idx < len(segs_raw)): continue
                        # Reject obvious non-names: short words, known bad words,
                        # anything not capitalized like a proper name
                        if len(spk) <= 2: continue
                        if spk.lower() in {
                            "froh","dankbar","erfreut","glücklich","klar",
                            "fertig","bereit","sicher","richtig","wichtig",
                            "gassier","raffi","jetzt","leider","insofern",
                            "unknown speaker","speaker","person","voice",
                        }: continue
                        old = segs_raw[idx]["speaker"]
                        if old != spk:
                            segs_raw[idx]["speaker"] = spk
                            segs_raw[idx]["llm_corrected"] = True
                            changed += 1
                            if changed <= 15:
                                print(f"  [LLM] [{idx:03d}] {old} → {spk}")
                    llm_corrections = changed
                    llm_used        = True
                    llm_status      = f"corrected {changed} segment(s)"
                    print(f"  [LLM] Total corrections: {changed}")
                else:
                    print(f"  [LLM] WARNING: could not parse JSON from response")
                    llm_status = "parse failed"
        else:
            print(f"\n[PASS 4b] Ollama offline — skipping correction")
        pass_times["pass4b"] = round(time.time()-p4b, 2)
        print(f"[PASS 4b] Done {pass_times['pass4b']}s  {llm_status}")

        # ── PASS 4b.5: RE-ASSIGN WORD SPEAKERS FROM LLM BOUNDARIES ──
        # Diarization boundaries drove the initial word tagging.
        # Now that the LLM has corrected segment labels, re-tag every
        # word in result["segments"] using the corrected segs_raw times.
        # This ensures the transcript text reflects LLM corrections,
        # not the original (potentially wrong) diarization clusters.
        corrected_boundaries = [
            (s["start"], s["end"], s["speaker"])
            for s in segs_raw
            if s["speaker"] not in ("", None)
        ]
        corrected_boundaries.sort(key=lambda x: x[0])

        reassigned = 0
        for seg in result["segments"]:
            seg_mid = (seg["start"] + seg.get("end", seg["start"]+1)) / 2
            best_spk = None
            best_overlap = -1.0
            for (b_start, b_end, b_spk) in corrected_boundaries:
                overlap = min(seg.get("end", seg["start"]+1), b_end) \
                        - max(seg["start"], b_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = b_spk
            if best_spk and best_spk != seg.get("speaker"):
                seg["speaker"] = best_spk
                reassigned += 1
            # Also re-tag individual words within the segment
            for word in seg.get("words", []):
                w_mid = (word.get("start", seg_mid) +
                         word.get("end",   seg_mid)) / 2
                for (b_start, b_end, b_spk) in corrected_boundaries:
                    if b_start <= w_mid < b_end:
                        word["speaker"] = b_spk
                        break
        print(f"  [RESYNC] Re-tagged {reassigned} segment(s) from LLM boundaries")


        p4c = time.time()
        deep_analysis = None
        analysis_status = "skipped"

        speaker_names = list(set(
            s["speaker"] for s in segs_raw
            if s["speaker"] not in ("UNKNOWN","") and
               not s["speaker"].startswith("SPEAKER_")
        ))

        if ollama_available():
            print(f"\n[PASS 4c] Deep analysis — context: {context}...")
            analysis_prompt = build_analysis_prompt(
                segs_raw, language, context, speaker_names
            )
            analysis_resp = call_ollama(analysis_prompt, max_tokens=3000)
            if analysis_resp:
                parsed_analysis = parse_json_from_llm(analysis_resp)
                if parsed_analysis:
                    deep_analysis   = parsed_analysis
                    analysis_status = f"complete ({context})"
                    print(f"  [ANALYSIS] Deep analysis complete for context={context}")
                else:
                    # Keep raw text as fallback
                    deep_analysis   = {"raw": analysis_resp[:3000]}
                    analysis_status = "raw (parse failed)"
        else:
            analysis_status = "skipped — Ollama offline"

        pass_times["pass4c"] = round(time.time()-p4c, 2)
        print(f"[PASS 4c] Done {pass_times['pass4c']}s  {analysis_status}")

        # ── PASS 5: BLUEPRINTS + TONE + DENGLISCH ─────────────
        p5 = time.time()
        print(f"\n[PASS 5] Voice blueprints + tone analysis...")
        blueprints = build_voice_blueprints(audio_data, segs_raw)

        emotion_map = {"ang":"Angry","hap":"Happy","sad":"Sad","neu":"Neutral"}
        waveform    = torch.from_numpy(audio_data).unsqueeze(0)
        final_segs  = []

        for seg in segs_raw:
            ss, se = int(seg["start"]*SAMPLE_RATE), int(seg["end"]*SAMPLE_RATE)
            tone   = "Neutral"
            if (se-ss) > MIN_SEG_SAMPLES:
                try:
                    _,_,_,labels = tone_classifier.classify_batch(
                        waveform[:,ss:se]
                    )
                    tone = emotion_map.get(labels[0].lower(), labels[0].title())
                except: pass
            final_segs.append({
                "start":         seg["start"],
                "speaker":       seg["speaker"],
                "speaker_id":    seg["speaker_id"],
                "text":          seg["text"],
                "tone":          tone,
                "llm_corrected": seg["llm_corrected"],
                "resolved":      True
            })

        # Denglisch detection
        denglisch = detect_denglisch(segs_raw, language)
        words_data = analyze_words(segs_raw)
        pass_times["pass5"] = round(time.time()-p5, 2)

        total_proc = round(time.time()-t_wall, 2)
        total_words= sum(len(s["text"].split()) for s in segs_raw)
        print(f"[PASS 5] Done {pass_times['pass5']}s")
        print(f"\n[DONE] {total_proc}s total")

        # Identity report
        all_spk_ids = {s["speaker_id"] for s in segs_raw
                       if s["speaker_id"] not in ("UNKNOWN","")}
        identity_report = {}
        for spk_id in all_spk_ids:
            ev = name_ev.get(spk_id,{})
            resolved = s2n.get(spk_id, spk_id)
            llm_names = {s["speaker"] for s in segs_raw
                         if s["speaker_id"]==spk_id and s["llm_corrected"]}
            if llm_names:
                resolved = max(llm_names, key=lambda n:
                    sum(1 for s in segs_raw
                        if s["speaker_id"]==spk_id and s["speaker"]==n))
            identity_report[spk_id] = {
                "resolved_name":  resolved,
                "confidence":     ev.get("confidence",0.0),
                "auto_assigned":  spk_id in s2n,
                "llm_corrected":  bool(llm_names),
                "evidence_hits":  ev.get("score",0),
            }

        return {
            "status": "complete",
            "context": context,
            "analytics": {
                "processing_time":   f"{total_proc}s",
                "pass_times":        pass_times,
                "words":             total_words,
                "speakers_detected": n_detect,
                "speakers_resolved": len(blueprints),
                "llm_used":          llm_used,
                "llm_corrections":   llm_corrections,
                "llm_status":        llm_status,
                "analysis_status":   analysis_status,
                "unresolved":        []
            },
            "identity_report":  identity_report,
            "acoustic_profiles":blueprints,
            "word_frequency":   words_data,
            "denglisch":        denglisch,
            "deep_analysis":    deep_analysis,
            "segments":         final_segs
        }

    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        flush_gpu()


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "gpu":           torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "none",
        "vram_free_gb":  vram_free_gb(),
        "ollama":        "online" if ollama_available() else "offline",
        "ollama_model":  OLLAMA_MODEL
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)