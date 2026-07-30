"""
Tools the recommender agent can call during its reasoning chain.

Each function is a small, pure "tool" with a clear name, typed inputs, and a
structured return value. The agent (src/agent.py) decides *which* tool to call
at each step and records the call + result in its reasoning trace, the same way
an LLM agent emits tool-calls and reads back observations.

Keeping these as standalone, side-effect-free functions makes them easy to test
in isolation (tests/test_tools.py) and easy to log.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Songs with acousticness above this are treated as "acoustic".
ACOUSTIC_THRESHOLD = 0.6
ENERGY_MIN, ENERGY_MAX = 0.0, 1.0

# Moods that clash with a high-energy request (used by detect_preference_conflict).
LOW_ENERGY_MOODS = {"sad", "melancholy", "relaxed", "chill"}
CONFLICT_ENERGY = 0.7


def catalog_stats(songs: List[Dict]) -> Dict:
    """TOOL: summarise the catalog the agent is working with.

    Returns counts, the set of genres/moods available, and the energy span, so
    the agent can reason about what is even possible before recommending.
    """
    genres: Dict[str, int] = {}
    moods: Dict[str, int] = {}
    energies: List[float] = []
    for s in songs:
        genres[s["genre"]] = genres.get(s["genre"], 0) + 1
        moods[s["mood"]] = moods.get(s["mood"], 0) + 1
        energies.append(float(s["energy"]))
    return {
        "n_songs": len(songs),
        "n_genres": len(genres),
        "genre_counts": genres,
        "mood_counts": moods,
        "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
    }


def validate_energy(value: object) -> Dict:
    """TOOL: coerce and range-check an energy value.

    Returns {value, changed, message} — the sanitised value on the 0-1 scale,
    whether it had to be changed, and a human-readable note if so.
    """
    try:
        energy = float(value)
    except (TypeError, ValueError):
        return {"value": 0.5, "changed": True,
                "message": f"energy {value!r} not numeric; using 0.5"}
    clamped = max(ENERGY_MIN, min(ENERGY_MAX, energy))
    if clamped != energy:
        return {"value": clamped, "changed": True,
                "message": f"energy {energy} out of range; clamped to {clamped}"}
    return {"value": clamped, "changed": False, "message": ""}


def detect_preference_conflict(prefs: Dict) -> Dict:
    """TOOL: flag internally contradictory requests.

    Example: a low-energy mood ("sad") paired with a high target energy. Returns
    {conflict: bool, reason: str}.
    """
    mood = prefs.get("mood")
    energy = prefs.get("energy")
    if (mood in LOW_ENERGY_MOODS
            and isinstance(energy, (int, float))
            and energy > CONFLICT_ENERGY):
        return {"conflict": True,
                "reason": f"mood '{mood}' usually implies low energy, but "
                          f"energy={energy} is high"}
    return {"conflict": False, "reason": ""}


def count_genres(recommendations: List[Tuple[Dict, float, str]]) -> Dict[str, int]:
    """TOOL: count how many times each genre appears in a shortlist.

    Used by the agent's CHECK step to detect a genre-dominated result set.
    """
    counts: Dict[str, int] = {}
    for song, _score, _reason in recommendations:
        counts[song["genre"]] = counts.get(song["genre"], 0) + 1
    return counts


# Registry so the agent (and the trace) can refer to tools by stable names.
TOOLS = {
    "catalog_stats": catalog_stats,
    "validate_energy": validate_energy,
    "detect_preference_conflict": detect_preference_conflict,
    "count_genres": count_genres,
}
