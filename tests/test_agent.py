"""Tests for the agentic plan/act/check/fix workflow."""

import pytest

from src.agent import AgenticRecommender, MAX_ITERATIONS, MAX_SAME_GENRE


def _catalog():
    """A catalog dominated by 'pop' so a naive scorer returns a monolithic set,
    forcing the agent's diversity FIX to kick in."""
    base = dict(tempo_bpm=120, valence=0.7, acousticness=0.2)
    songs = []
    for i in range(6):
        songs.append(dict(
            id=i, title=f"Pop {i}", artist="A", genre="pop", mood="happy",
            energy=0.8, danceability=0.8, **base,
        ))
    # Several non-pop songs of distinct genres so a diverse shortlist is
    # actually achievable once the agent penalises repeated genres.
    others = [
        ("rock", "intense"), ("lofi", "chill"), ("jazz", "relaxed"),
        ("edm", "energetic"), ("indie", "moody"),
    ]
    for j, (genre, mood) in enumerate(others):
        songs.append(dict(id=90 + j, title=f"{genre} {j}", artist="B",
                          genre=genre, mood=mood, energy=0.7,
                          danceability=0.6, **base))
    return songs


def test_agent_returns_k_recommendations():
    agent = AgenticRecommender(_catalog(), k=5)
    result = agent.run({"genre": "pop", "mood": "happy", "energy": 0.8})
    assert len(result.recommendations) == 5
    assert result.iterations >= 1


def test_agent_enforces_diversity_via_fix_loop():
    """With a pop-heavy catalog the first pass over-picks 'pop'; the CHECK stage
    should flag it and a FIX iteration should reduce genre repetition."""
    agent = AgenticRecommender(_catalog(), k=5)
    result = agent.run({"genre": "pop", "mood": "happy", "energy": 0.8})

    genres = [s["genre"] for s, _, _ in result.recommendations]
    assert genres.count("pop") <= MAX_SAME_GENRE
    # It should have needed at least one fix loop to get there.
    assert result.iterations >= 2
    assert result.strategy["diversity_penalty"] > 0.0


def test_agent_clamps_out_of_range_energy():
    agent = AgenticRecommender(_catalog(), k=3)
    result = agent.run({"genre": "pop", "mood": "happy", "energy": 5.0})
    assert any("clamped" in w for w in result.warnings)


def test_agent_handles_empty_prefs_safely():
    agent = AgenticRecommender(_catalog(), k=3)
    result = agent.run({})
    assert len(result.recommendations) == 3
    assert any("empty preferences" in w for w in result.warnings)


def test_agent_respects_iteration_cap():
    agent = AgenticRecommender(_catalog(), k=3)
    result = agent.run({"genre": "pop", "mood": "happy", "energy": 0.8})
    assert result.iterations <= MAX_ITERATIONS


def test_agent_rejects_empty_catalog():
    with pytest.raises(ValueError):
        AgenticRecommender([], k=3)


def test_plan_does_not_mutate_caller_prefs():
    agent = AgenticRecommender(_catalog(), k=3)
    prefs = {"genre": "pop", "mood": "happy", "energy": 5.0}
    agent.run(prefs)
    assert prefs["energy"] == 5.0  # caller's dict untouched
