# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**IntelliTunes** — a self-correcting music recommender.

---

## 2. Intended Use

VibeFinder recommends the top *k* songs from a small catalog that best match a
user's stated taste profile (favorite genre, mood, target energy, danceability,
and acoustic preference). Each recommendation comes with a plain-language
explanation of why it was picked. It also has an evaluation harness that runs the agent over taste profiles and a score summary as well as a CI check.

- **What it generates:** a short, ranked shortlist of songs plus a per-song
  reason breakdown and a trace of how the system arrived at that list.
- **Assumptions about the user:** that the user can describe their taste as a few
  simple attributes, and that those attributes are stable for a single request.
  It assumes nothing about listening history, time of day, or context.
- **Who it's for:** classroom exploration and learning about how recommenders
  turn data into predictions. It is **not** built for real end users — the
  catalog is tiny, the taste model is simplistic, and it has no personalization
  over time.

---

## 3. How the Model Works

Think of it like a helpful friend picking songs for you. First you tell the
friend what you're into — say "pop, happy, high energy." The friend looks at each
song and gives it points: a point boost if the genre matches, a bigger boost if
the mood matches (mood captures the "vibe" best, so it's worth the most), and
partial points for how close a song's energy and danceability are to what you
asked for. Add a small bonus if the song's acoustic-ness matches your taste. Add
up the points and the highest scorers are the recommendations.

What makes this version different from a plain scorer is that the friend
**checks their own work before handing you the list**. After making a first pick
list, it asks itself two questions: "Is my top pick actually a strong match?" and
"Did I accidentally pick almost all one genre?" If something looks off, it
adjusts its approach — for example, it penalizes repeating the same genre — and
tries again, up to a few times. It also cleans up messy requests first: if you
ask for an impossible energy level like 5 (the scale only goes 0 to 1), it
gently corrects it and tells you it did.

Behind the scenes the "friend" reasons in steps and **calls small tools** to do
it — one tool summarizes the catalog, one checks the energy value is sensible,
one spots contradictory requests, and one counts how many songs share a genre.
Each thought, tool call, and decision is written down so you can read exactly how
the list was chosen.

**Changes from the starter logic:** the starter scored every song once and
printed the result. This version wraps that scoring in a **plan → act → check →
fix loop** driven by tool-calls, and adds a runtime diversity penalty and input
guardrails, so the final output can differ from the naive first pass.

---

## 4. Data

- **Size:** 17 songs in `data/songs.csv`.
- **Fields per song:** id, title, artist, genre, mood, energy, tempo_bpm,
  valence, danceability, acousticness.
- **Genres represented:** pop, indie pop, lofi, rock, jazz, ambient, synthwave,
  hip-hop, classical, reggae, edm, country, metal, r&b.
- **Moods represented:** happy, chill, intense, relaxed, focused, moody,
  confident, melancholy, uplifting, nostalgic, aggressive, romantic, energetic.
- **Changes:** the dataset is the provided starter catalog; no songs were added
  or removed.
- **What's missing:** the catalog is tiny and hand-made, so it under-represents
  most real-world genres (no folk, funk, k-pop, afrobeats, regional/non-Western
  music, etc.). It reflects the taste and vocabulary of whoever authored the
  seed data, not a broad or representative population of listeners.

---

## 5. Strengths

- **Transparent and auditable:** every recommendation shows which points it
  earned, and the agent records its full tool-call reasoning chain (saved to
  `logs/agent_trace.md`) — nothing is a black box.
- **Works well for clear, self-consistent profiles.** For example, a
  `pop / happy / high-energy` request returns "Sunrise City" as the obvious top
  pick with a strong score, matching intuition on the first pass.
- **Self-correcting on diversity.** For a `lofi / chill` request, the naive pass
  returns three lofi tracks in a row; the agent notices this and re-ranks so the
  shortlist keeps the best lofi picks but adds variety — a genuinely better list.
- **Fails safe.** Nonsense or out-of-range input (e.g. `energy = 5.0`, or empty
  preferences) is cleaned up and flagged rather than crashing the program.

---

## 6. Limitations and Bias

- **Features it ignores:** lyrics, language, tempo (beyond storage), artist
  familiarity, popularity, recency, and any notion of listening history or
  context. It cannot understand *why* a person likes something.
- **Mood is weighted highest (3.0),** so the system can over-index on mood and
  surface a mood-matching song even when the genre is a poor fit. This is a
  deliberate design bias that won't suit every user.
- **Small-catalog bias:** with only 17 songs and some genres appearing just once,
  requests for a rare genre have almost nothing to choose from, so results lean
  on the always-on energy/danceability points instead of real matches.
- **One-size taste shape:** the model treats every user with the same fixed
  weighting scheme. A user who cares far more about genre than mood is served the
  same scoring rule as everyone else.
- **Fairness angle:** because the data reflects one author's taste and genre
  vocabulary, listeners whose tastes fall outside that vocabulary are
  systematically served worse — a small-scale version of a real recommender
  under-serving underrepresented groups.

---

## 7. Evaluation

- **Profiles tested:**
  - `pop / happy / energy 0.8` — expected an obvious pop pick on top; got
    "Sunrise City" (score 9.03), passing the checks on the first iteration.
  - `lofi / chill / energy 0.4` — expected the recommender to lean too hard into
    lofi; it did, then self-corrected via the diversity fix.
  - `energy 5.0` (out of range) and empty preferences `{}` — expected graceful
    handling; both were clamped/defaulted with a visible warning and no crash.
- **What I looked for:** whether the top pick made intuitive sense, whether the
  list was overly repetitive, and whether bad input broke anything.
- **What surprised me:** how easily the scorer produced an all-one-genre list for
  a niche request — it made the value of an automatic diversity check obvious.
- **Evaluation harness:** `python3 -m src.evaluate` runs the agent over 5
  predefined profiles, checks each result against an expectation (correct top
  pick, diversity enforced, out-of-range clamped, empty input handled, conflict
  flagged), and prints a pass/fail score — currently **5/5 (100%)**.
- **Unit tests:** an automated suite (`python3 -m pytest`, 19 tests) covering the
  scoring logic, the tools, and the agent loop — diversity self-correction,
  energy clamping, empty-input safety, the iteration cap, the recorded reasoning
  trace, and that the agent never mutates the caller's input.

---

## 8. Future Work

- **Richer taste input:** let users weight what matters to them (e.g. "genre
  matters more than mood") instead of a single fixed weighting for everyone.
- **Use more features:** factor in tempo ranges, valence, and artist variety so
  the shortlist isn't dominated by one artist or a narrow BPM band.
- **Better explanations:** turn the point breakdown into a natural-language
  sentence ("Chosen because it's an upbeat pop track close to your energy level").
- **Smarter diversity:** balance diversity across mood and artist too, not just
  genre, and make the trade-off between relevance and variety user-adjustable.
- **Bigger, fairer catalog:** expand the dataset across many more genres and
  cultural traditions to reduce the built-in taste bias.

---

## 9. Personal Reflection

Building this made concrete how a recommender is really just a scoring rule plus
a lot of judgment calls hidden inside the weights — deciding mood is worth 3.0
and genre 2.0 quietly shapes everything the user sees. The most interesting part
was adding the self-checking loop: watching the system produce an all-lofi list,
notice the problem itself, and fix it felt like the difference between a
calculator and something that actually reasons about its own output, even with
simple rules. It also changed how I think about apps like Spotify — their
"magic" is layered on top of the same basic idea, and the choices about what to
measure and what to ignore are where bias sneaks in. Even when a model seems
smart, human judgment still decides what "a good recommendation" even means.


## Reflection and Ethics

Some limitations are the lack of genres. Some biases are that it is heavily reliant on mood.

It could be misused to create false assumptions

It is pretty reliable only when you know the context of what code the AI is writing as well as the overarching scope of the project as it tends to go off-task giving it generic prompts. 

I learned a lot about good practical habits using AI and problem-solving. I used it to analyze the project stucture first. One helpful AI suggestion implementing a guardrail that that prevents an infinite loop. However, one suggestion that was flawed is that I do feel that mood is sometimes over-weighted in the scoring formula and that it should not be generalized to every user that mood is the heaviest weighted metric for them. 