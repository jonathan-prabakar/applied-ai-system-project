# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders



The original name of this system is VibeFinder. This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users. 


**Title and Summary of new Agenticproject + Test Harness extensions**
The name of the new system is IntelliTunes. This system recommends the top *k* songs from a small catalog by scoring each song against a user's taste profile. Each song carries a set of features (genre, mood, energy, tempo, valence, danceability, acousticness) and the user profile carries preferences (favorite genre, mood, target energy, etc.). Every recommendation comes with a plain-language explanation of *why* it matched.

**What makes this version agentic:** the recommender no longer scores the
catalog once and prints whatever falls out. It runs a **PLAN → ACT → CHECK →
FIX** loop (an autonomous agent that *plans, acts, and checks its own work*):

1. **PLAN** — inspect and sanitise the request (clamp out-of-range values,
   detect empty or conflicting preferences) and choose a starting strategy.
2. **ACT** — produce a candidate recommendation set with that strategy.
3. **CHECK** — grade its own output against quality guardrails (is the top pick
   relevant enough? is the shortlist too dominated by one genre? did anything
   invalid slip through?). This is the agent *testing its own work*.
4. **FIX** — if a check fails, adjust the strategy (raise the diversity penalty,
   sharpen genre/mood weights) and loop back to ACT.

This mirrors a coding assistant that writes code, runs the tests, reads the
failures, and edits until the tests pass — here the "tests" are recommendation
quality checks and the "edits" are strategy adjustments. The loop is bounded by
a hard iteration cap, every step is logged, and bad input is handled safely
instead of crashing. The agent lives in [src/agent.py](src/agent.py) and is the
default path used by [src/main.py](src/main.py). See
[diagrams/architecture.mmd](diagrams/architecture.mmd) for the full flow.

**Tool-calling reasoning chain.** Each stage reasons in steps by calling small,
named tools in [src/tools.py](src/tools.py) — `catalog_stats`, `validate_energy`,
`detect_preference_conflict`, and `count_genres` — reading back the result, and
deciding what to do next. Every `thought → tool call → observation → decision`
is recorded and can be saved to a committed log; see
[ai_interactions.md](ai_interactions.md) and [logs/agent_trace.md](logs/agent_trace.md).

**Requirements:** Python 3.8+ and the packages in
[requirements.txt](requirements.txt) (`pandas`, `pytest`, `streamlit`). The core
recommender and agent use only the Python standard library — no API keys or
network access are needed.
1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app **from the project root** (so `data/songs.csv` resolves):

   ```bash
   python3 -m src.main                 # normal run: prints the agent trace + picks
   python3 -m src.main --verbose       # also stream the agent's PLAN/ACT/CHECK/FIX logs
   python3 -m src.main -k 3            # ask for a different number of songs
   python3 -m src.main --trace-out logs/agent_trace.md   # save the full reasoning chain
   ```

   You'll see an **AGENT WORKFLOW TRACE** showing each plan/act/check/fix step,
   the final recommendations with reasons, and any guardrail notes.

### Running Tests

```bash
python3 -m pytest        # or simply: pytest  (19 tests)
```

- [tests/test_recommender.py](tests/test_recommender.py) covers the scoring logic.
- [tests/test_agent.py](tests/test_agent.py) covers the agentic loop: diversity
  self-correction, out-of-range clamping, empty-input safety, and the iteration cap.
- [tests/test_tools.py](tests/test_tools.py) covers the callable tools and the
  recorded reasoning trace.

### Evaluation Harness

Run the system on a set of predefined profiles and print a pass/fail score:

```bash
python3 -m src.evaluate            # summary score (exits non-zero if any fail)
python3 -m src.evaluate --verbose  # show the detail + iteration count per scenario
```

[src/evaluate.py](src/evaluate.py) checks correct top pick, diversity enforcement,
out-of-range clamping, empty-input handling, and conflict detection — currently
**5/5 (100%)**.

---

## Sample Interactions

Three real runs against `data/songs.csv` (17 songs). Traces are copied verbatim
from the program output.

### Example A — clean request, passes on the first try

**Input:** `{genre: "pop", mood: "happy", energy: 0.8}`, k=4

```
PLAN: strategy=weights(g=2.0,m=3.0,e=2.0,d=1.5,a=1.0) diversity=0.0
ACT #1: 4 recs, top score 9.03
CHECK #1: PASSED
-> settled after 1 iteration(s)

1. Sunrise City — Neon Echo       (pop/happy)         9.03
2. Rooftop Lights — Indigo Parade (indie pop/happy)   6.94
3. Gym Hero — Max Pulse           (pop/intense)       5.67
4. Iron Verdict — Blacklung Order (metal/aggressive)  4.13
```

The catalog already has a strong, varied match set, so the agent's CHECK stage
finds no issues and stops after one iteration.

### Example B — the agent corrects its own work (the key demo)

**Input:** `{genre: "lofi", mood: "chill", energy: 0.4}`, k=5

```
PLAN: strategy=weights(g=2.0,m=3.0,e=2.0,d=1.5,a=1.0) diversity=0.0
ACT #1: 5 recs, top score 8.28
CHECK #1: 1 issue(s) -> genre 'lofi' appears 3x (> 2)
FIX #1: raised diversity_penalty to 1.5
ACT #2: 5 recs, top score 8.28
CHECK #2: PASSED
-> settled after 2 iteration(s)

1. Library Rain — Paper Lanterns       (lofi/chill)        8.28
2. Midnight Coding — LoRoom             (lofi/chill)        8.28
3. Spacewalk Thoughts — Orbit Bloom     (ambient/chill)     6.12
4. Velvet Hours — Sable Rose            (r&b/romantic)      3.85
5. Island Time — Coral Sound System     (reggae/uplifting)  3.69
```

The first pass returned three `lofi` tracks. The agent **caught this itself** in
CHECK, applied a diversity penalty in FIX, and re-ran — the second shortlist keeps
the two best lofi picks but breaks up the genre block. This is the system
*changing its own behavior based on self-assessment*, not a fixed ranking.

### Example C — a guardrail handles bad input safely

**Input:** `{genre: "pop", mood: "happy", energy: 5.0}` (energy is out of the 0–1 range), k=3

```
PLAN: strategy=weights(g=2.0,m=3.0,e=2.0,d=1.5,a=1.0) diversity=0.0
PLAN: guardrail note - energy 5.0 out of range; clamped to 1.0
ACT #1: 3 recs, top score 8.71
CHECK #1: PASSED

GUARDRAIL NOTES
! energy 5.0 out of range; clamped to 1.0

1. Sunrise City   (pop)        8.71
2. Rooftop Lights (indie pop)  6.54
3. Gym Hero       (pop)        5.79
```

Instead of crashing or silently trusting bad data, PLAN clamps the value to a
valid range, records a warning the user can see, and continues.

---

## Design Decisions

- **Rule-based agent, not an LLM call.** The plan/act/check/fix loop is
  deterministic. *Trade-off:* it's less "smart" than delegating to an LLM, but it
  runs offline with no API key, is fully reproducible, is free, and every decision
  is inspectable in the trace — which matters for a classroom/eval project.
- **Integrated into the main path, not a standalone script.** `src/main.py` runs
  the agent by default, and the agent tunes the *actual* scoring knobs
  (`weights` and `diversity_penalty` in [src/recommender.py](src/recommender.py))
  at runtime. So the recommendations genuinely differ from the old single-pass
  scorer. *Trade-off:* I had to refactor `recommend_songs` to accept weight and
  diversity overrides — done with backward-compatible defaults so existing
  callers and tests are unaffected.
- **Guardrails over cleverness.** A hard `MAX_ITERATIONS` cap prevents an
  infinite fix loop; input clamping and an empty-preferences fallback prevent
  crashes; Python's `logging` records every PLAN/ACT/CHECK/FIX step. *Trade-off:*
  when an issue can't be resolved within the cap, the agent returns its best
  effort and logs that it did so, rather than looping forever.
- **Diversity as a greedy re-rank penalty, not a hard quota.** When one genre
  dominates, the agent penalizes repeated genres during selection instead of
  imposing a fixed "max N per genre" rule. This keeps the highest-relevance picks
  while still breaking up monotonous results.

---

## Testing Summary

Run with `python3 -m pytest` — **9 tests pass** (2 scoring + 7 agent).

**What worked:** The agent tests confirm the behaviors that matter — diversity
self-correction actually fires and reduces genre repetition, out-of-range energy
is clamped, empty preferences are handled without crashing, the iteration cap is
respected, and the agent never mutates the caller's input dict.

**What didn't (at first):** My initial diversity test fixture had only two
non-`pop` songs, so a shortlist of five with "at most two pop" was
*mathematically impossible* — the test failed even though the code was correct.
I fixed the **fixture** (added five distinct non-pop genres), not the loop.

**What I learned:** The diversity penalty has to *escalate across iterations* to
overcome the head start that genre + mood matches give a song. A single small
penalty wasn't enough to dislodge a dominant genre, which is exactly why the FIX
step increments the penalty on each loop rather than setting it once.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---


## How The System Works

**Song features:** genre, mood, energy, tempo_bpm, valence, danceability,
acousticness.

**User profile:** favorite_genre, favorite_mood, target_energy,
target_danceability, likes_acoustic.

**Scoring** (in [src/recommender.py](src/recommender.py)) — each song earns a
weighted score:

- genre match: +2.0 (flat, exact category match)
- mood match: +3.0 (mood best captures "vibe", so it's weighted highest)
- energy: up to +2.0, scaled by how *close* the song is to the target
- danceability: up to +1.5, same closeness idea
- acoustic preference match: +1.0

The top *k* songs by score are recommended, each with an explanation of the
points it earned.

**The agentic layer** ([src/agent.py](src/agent.py)) wraps this scorer in a
self-correcting loop. Rather than trusting the first pass, it CHECKs the result
and, if a guardrail trips, it re-runs with an adjusted strategy:

| Guardrail (CHECK) | What it catches | FIX applied |
|---|---|---|
| Relevance | Top score below the quality bar | Boost genre/mood weights |
| Diversity | One genre dominates the shortlist | Raise the diversity penalty and re-rank |
| Input validity (PLAN) | Out-of-range / empty / conflicting prefs | Clamp values, soften weights, record a warning |

Because scoring weights and a diversity penalty are now *tuned at runtime by the
agent*, the system's output genuinely changes based on its own self-assessment —
not just on the fixed weights.

## Reflection
I learned a lot about good practical habits using AI and problem-solving. I used it to analyze the project stucture first. One helpful AI suggestion implementing a guardrail that that prevents an infinite loop. However, one suggestion that was flawed is that I do feel that mood is sometimes over-weighted in the scoring formula and that it should not be generalized to every user that mood is the heaviest weighted metric for them. 





## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

