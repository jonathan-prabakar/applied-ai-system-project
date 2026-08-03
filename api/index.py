"""
FastAPI web layer for the IntelliTunes agentic music recommender.

Serves a browser UI at "/" (GET) and a JSON API at "/recommend" (POST).
The recommender logic in src/ is reused as-is — this file only adds the
HTTP/HTML surface so the project can be deployed to Vercel.

Run locally:
    uvicorn api.index:app --reload
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import AgenticRecommender
from src.recommender import load_songs

app = FastAPI(title="IntelliTunes Agentic Recommender", version="1.0.0")

# ---- Load the catalog once at startup --------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SONGS_PATH = os.path.join(DATA_DIR, "songs.csv")


def _load_catalog() -> List[Dict]:
    for candidate in (SONGS_PATH, "data/songs.csv"):
        if os.path.exists(candidate):
            songs = load_songs(candidate)
            if songs:
                return songs
    raise RuntimeError(f"Could not load song catalog (tried {SONGS_PATH} and data/songs.csv)")


SONGS = _load_catalog()

# Unique genres/moods from the catalog for the UI dropdowns.
GENRES = sorted({s["genre"] for s in SONGS})
MOODS = sorted({s["mood"] for s in SONGS})

# ---- Request/response models -----------------------------------------------
class RecommendRequest(BaseModel):
    genre: str = Field(default="pop", description="Favorite genre")
    mood: str = Field(default="happy", description="Favorite mood")
    energy: float = Field(default=0.5, ge=0.0, le=1.0, description="Target energy 0-1")
    danceability: float = Field(default=0.5, ge=0.0, le=1.0, description="Target danceability 0-1")
    likes_acoustic: bool = Field(default=False, description="Prefers acoustic songs")
    k: int = Field(default=5, ge=1, le=10, description="How many songs to recommend")


def _result_to_payload(result) -> Dict[str, Any]:
    """Convert an AgentResult into a JSON-serialisable dict."""
    recommendations = []
    for rank, (song, score, explanation) in enumerate(result.recommendations, start=1):
        recommendations.append({
            "rank": rank,
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "energy": song["energy"],
            "score": round(score, 2),
            "explanation": explanation,
        })
    return {
        "recommendations": recommendations,
        "iterations": result.iterations,
        "trace": result.trace,
        "warnings": result.warnings,
        "strategy": result.strategy,
    }


# ---- API routes -------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "songs_loaded": len(SONGS)}


@app.get("/catalog")
def catalog() -> Dict[str, Any]:
    """Expose the available genres/moods so the UI can populate its dropdowns."""
    return {
        "genres": GENRES,
        "moods": MOODS,
        "default": {"genre": "pop", "mood": "happy", "energy": 0.6},
        "n_songs": len(SONGS),
    }


@app.post("/recommend")
def recommend(body: RecommendRequest) -> Dict[str, Any]:
    prefs = {
        "genre": body.genre,
        "mood": body.mood,
        "energy": body.energy,
        "danceability": body.danceability,
        "likes_acoustic": body.likes_acoustic,
    }
    try:
        agent = AgenticRecommender(SONGS, k=body.k)
        result = agent.run(prefs)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Recommender failed: {exc}")
    return _result_to_payload(result)


# ---- Browser UI -------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IntelliTunes — Agentic Music Recommender</title>
<style>
  :root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card2: #222244;
    --text: #e8e8f0;
    --muted: #9a9ab0;
    --accent: #7c5cff;
    --accent2: #00d4aa;
    --warn: #ffb347;
    --danger: #ff6b6b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 20% -10%, #2a1a5e 0%, var(--bg) 55%);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.5;
    padding: 2rem 1rem;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 2rem; }
  header h1 {
    font-size: 2.4rem;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
  }
  header p { color: var(--muted); margin-top: 0.4rem; }
  .badge {
    display: inline-block; background: var(--card2); border: 1px solid #333;
    border-radius: 999px; padding: 0.2rem 0.8rem; font-size: 0.8rem;
    color: var(--muted); margin-top: 0.6rem;
  }
  .card {
    background: var(--card); border: 1px solid #2c2c4a; border-radius: 14px;
    padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  }
  .card h2 {
    font-size: 1.1rem; margin-bottom: 1rem; color: var(--accent2);
    display: flex; align-items: center; gap: 0.5rem;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
  label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.3rem; }
  select, input[type="number"], input[type="range"] {
    width: 100%; padding: 0.6rem 0.8rem; border-radius: 8px;
    border: 1px solid #3a3a5c; background: var(--card2); color: var(--text);
    font-size: 1rem; outline: none;
  }
  select:focus, input:focus { border-color: var(--accent); }
  input[type="range"] { padding: 0; accent-color: var(--accent); }
  .range-row { display: flex; align-items: center; gap: 0.8rem; }
  .range-val {
    min-width: 46px; text-align: center; font-weight: 600; color: var(--accent2);
    background: var(--card2); border-radius: 6px; padding: 0.2rem 0.4rem; font-size: 0.9rem;
  }
  .btn {
    width: 100%; margin-top: 1.2rem; padding: 0.85rem; font-size: 1.05rem;
    font-weight: 700; border: none; border-radius: 10px; cursor: pointer;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #fff; letter-spacing: 0.5px; transition: transform 0.15s, box-shadow 0.15s;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(124,92,255,0.4); }
  .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
  .rec {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 1rem; padding: 1rem 0; border-bottom: 1px solid #2c2c4a;
  }
  .rec:last-child { border-bottom: none; }
  .rank {
    flex-shrink: 0; width: 2rem; height: 2rem; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.9rem;
  }
  .rec-title { font-weight: 700; font-size: 1.05rem; }
  .rec-artist { color: var(--muted); font-size: 0.9rem; }
  .rec-meta { margin-top: 0.4rem; font-size: 0.82rem; color: var(--muted); }
  .rec-meta span {
    background: var(--card2); padding: 0.15rem 0.5rem; border-radius: 999px;
    margin-right: 0.4rem; display: inline-block; margin-bottom: 0.25rem;
  }
  .rec-score { flex-shrink: 0; font-weight: 800; font-size: 1.2rem; color: var(--accent2); }
  .rec-why { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }
  .trace { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
  .trace-step {
    padding: 0.45rem 0.7rem; border-left: 3px solid var(--accent); background: var(--card2);
    margin-bottom: 0.4rem; border-radius: 0 6px 6px 0;
  }
  .warn {
    color: var(--warn); font-size: 0.9rem; padding: 0.5rem 0.8rem;
    background: rgba(255,179,71,0.08); border-radius: 8px; margin-bottom: 0.5rem;
  }
  .error { color: var(--danger); }
  .hidden { display: none; }
  #loading { text-align: center; color: var(--muted); padding: 2rem; }
  .spin {
    display: inline-block; width: 2rem; height: 2rem; border: 3px solid var(--card2);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  details summary { cursor: pointer; color: var(--accent2); font-weight: 600; margin-top: 1rem; }
  footer { text-align: center; color: #5a5a78; font-size: 0.8rem; margin-top: 2rem; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🎧 IntelliTunes</h1>
    <p>Agentic music recommender — PLAN → ACT → CHECK → FIX</p>
    <span class="badge" id="catalog-badge">Catalog loading…</span>
  </header>

  <div class="card">
    <h2>🎛️ Your taste profile</h2>
    <form id="rec-form">
      <div class="grid">
        <div>
          <label for="genre">Favorite genre</label>
          <select id="genre" name="genre"></select>
        </div>
        <div>
          <label for="mood">Mood</label>
          <select id="mood" name="mood"></select>
        </div>
        <div>
          <label for="energy">Energy (0–1)</label>
          <div class="range-row">
            <input type="range" id="energy" name="energy" min="0" max="1" step="0.05" value="0.6">
            <span class="range-val" id="energy-val">0.60</span>
          </div>
        </div>
        <div>
          <label for="k">Number of songs (k)</label>
          <select id="k" name="k">
            <option value="3">3</option>
            <option value="5" selected>5</option>
            <option value="8">8</option>
          </select>
        </div>
      </div>
      <button type="submit" class="btn" id="submit-btn">✨ Get recommendations</button>
    </form>
  </div>

  <div id="loading" class="hidden">
    <div class="spin"></div>
    <p style="margin-top:0.6rem">Running the agent loop…</p>
  </div>

  <div class="card hidden" id="results-card">
    <h2>🎵 Top recommendations</h2>
    <div id="warnings"></div>
    <div id="recommendations"></div>

    <details>
      <summary>🤖 Agent workflow trace (PLAN → ACT → CHECK → FIX)</summary>
      <div class="trace" id="trace" style="margin-top:0.8rem"></div>
    </details>
  </div>

  <div class="card hidden error" id="error-card">
    <h2>⚠️ Something went wrong</h2>
    <p id="error-msg"></p>
  </div>

  <footer>IntelliTunes · classroom recommender · deployed on Vercel</footer>
</div>

<script>
const $ = (id) => document.getElementById(id);

// Populate dropdowns from the catalog endpoint.
async function loadCatalog() {
  try {
    const res = await fetch('/catalog');
    const data = await res.json();
    const genreSelect = $('genre');
    const moodSelect = $('mood');
    data.genres.forEach(g => {
      const opt = document.createElement('option');
      opt.value = g; opt.textContent = g;
      genreSelect.appendChild(opt);
    });
    data.moods.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      moodSelect.appendChild(opt);
    });
    genreSelect.value = data.default.genre;
    moodSelect.value = data.default.mood;
    $('energy').value = data.default.energy;
    $('energy-val').textContent = Number(data.default.energy).toFixed(2);
    $('catalog-badge').textContent = data.n_songs + ' songs in catalog';
  } catch (e) {
    $('catalog-badge').textContent = 'Failed to load catalog';
  }
}

// Live-update the energy value label.
$('energy').addEventListener('input', () => {
  $('energy-val').textContent = Number($('energy').value).toFixed(2);
});

// Submit the form and render results.
$('rec-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('submit-btn');
  btn.disabled = true;
  $('loading').classList.remove('hidden');
  $('results-card').classList.add('hidden');
  $('error-card').classList.add('hidden');

  const body = {
    genre: $('genre').value,
    mood: $('mood').value,
    energy: parseFloat($('energy').value),
    k: parseInt($('k').value, 10)
  };

  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');

    renderRecommendations(data);
    renderTrace(data);
    $('results-card').classList.remove('hidden');
  } catch (err) {
    $('error-msg').textContent = err.message;
    $('error-card').classList.remove('hidden');
  } finally {
    btn.disabled = false;
    $('loading').classList.add('hidden');
  }
});

function renderRecommendations(data) {
  const wrap = $('recommendations');
  const warnWrap = $('warnings');
  wrap.innerHTML = '';
  warnWrap.innerHTML = '';

  (data.warnings || []).forEach(w => {
    const div = document.createElement('div');
    div.className = 'warn';
    div.textContent = '⚠️ ' + w;
    warnWrap.appendChild(div);
  });

  if (!data.recommendations || data.recommendations.length === 0) {
    wrap.innerHTML = '<p style="color:var(--muted)">No recommendations returned.</p>';
    return;
  }

  data.recommendations.forEach(r => {
    const div = document.createElement('div');
    div.className = 'rec';

    const left = document.createElement('div');
    left.style.display = 'flex';
    left.style.gap = '0.8rem';
    left.style.alignItems = 'flex-start';

    const rank = document.createElement('div');
    rank.className = 'rank';
    rank.textContent = r.rank;

    const info = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'rec-title';
    title.textContent = r.title;
    const artist = document.createElement('div');
    artist.className = 'rec-artist';
    artist.textContent = r.artist;

    const meta = document.createElement('div');
    meta.className = 'rec-meta';
    [r.genre, r.mood, 'energy ' + r.energy].forEach(t => {
      const s = document.createElement('span');
      s.textContent = t;
      meta.appendChild(s);
    });

    const why = document.createElement('div');
    why.className = 'rec-why';
    why.textContent = r.explanation;

    info.appendChild(title);
    info.appendChild(artist);
    info.appendChild(meta);
    info.appendChild(why);
    left.appendChild(rank);
    left.appendChild(info);

    const score = document.createElement('div');
    score.className = 'rec-score';
    score.textContent = r.score.toFixed(2);

    div.appendChild(left);
    div.appendChild(score);
    wrap.appendChild(div);
  });
}

function renderTrace(data) {
  const wrap = $('trace');
  wrap.innerHTML = '';
  (data.trace || []).forEach(step => {
    const div = document.createElement('div');
    div.className = 'trace-step';
    div.textContent = step;
    wrap.appendChild(div);
  });
  const settled = document.createElement('div');
  settled.className = 'trace-step';
  settled.style.borderColor = 'var(--accent2)';
  settled.textContent = 'Settled after ' + data.iterations + ' iteration(s)';
  wrap.appendChild(settled);
}

loadCatalog();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_TEMPLATE