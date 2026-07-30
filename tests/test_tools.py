"""Tests for the agent's callable tools and its recorded reasoning chain."""

from src import tools
from src.agent import AgenticRecommender


def _catalog():
    base = dict(tempo_bpm=120, valence=0.7, acousticness=0.2)
    songs = [dict(id=i, title=f"Pop {i}", artist="A", genre="pop", mood="happy",
                  energy=0.8, danceability=0.8, **base) for i in range(4)]
    for j, (g, m) in enumerate([("rock", "intense"), ("lofi", "chill")]):
        songs.append(dict(id=90 + j, title=f"{g}", artist="B", genre=g, mood=m,
                          energy=0.6, danceability=0.6, **base))
    return songs


# ---- tools.py ----------------------------------------------------------

def test_catalog_stats_counts_genres():
    stats = tools.catalog_stats(_catalog())
    assert stats["n_songs"] == 6
    assert stats["genre_counts"]["pop"] == 4
    assert stats["n_genres"] == 3


def test_validate_energy_clamps_out_of_range():
    out = tools.validate_energy(5.0)
    assert out["value"] == 1.0 and out["changed"] is True
    assert "clamped" in out["message"]


def test_validate_energy_handles_non_numeric():
    out = tools.validate_energy("loud")
    assert out["value"] == 0.5 and out["changed"] is True


def test_validate_energy_passes_valid_value():
    out = tools.validate_energy(0.7)
    assert out["value"] == 0.7 and out["changed"] is False


def test_detect_preference_conflict_flags_sad_high_energy():
    out = tools.detect_preference_conflict({"mood": "sad", "energy": 0.9})
    assert out["conflict"] is True and "sad" in out["reason"]


def test_detect_preference_conflict_ok_for_consistent_request():
    out = tools.detect_preference_conflict({"mood": "happy", "energy": 0.9})
    assert out["conflict"] is False


def test_count_genres_counts_shortlist():
    recs = [({"genre": "pop"}, 5.0, ""), ({"genre": "pop"}, 4.0, ""),
            ({"genre": "rock"}, 3.0, "")]
    assert tools.count_genres(recs) == {"pop": 2, "rock": 1}


# ---- reasoning chain recorded on AgentResult ---------------------------

def test_agent_records_tool_calls_in_reasoning():
    result = AgenticRecommender(_catalog(), k=3).run(
        {"genre": "pop", "mood": "happy", "energy": 0.8})
    called = {step.tool for step in result.reasoning if step.tool}
    # PLAN and CHECK should have exercised these tools.
    assert {"catalog_stats", "validate_energy",
            "detect_preference_conflict", "count_genres"} <= called
    # Every recorded step names a stage.
    assert all(step.stage in {"PLAN", "ACT", "CHECK", "FIX"}
               for step in result.reasoning)


def test_to_markdown_renders_steps():
    result = AgenticRecommender(_catalog(), k=3).run(
        {"genre": "pop", "mood": "happy", "energy": 5.0})
    md = result.to_markdown(header="Test run")
    assert "Test run" in md
    assert "Tool call" in md
    assert "clamped to 1.0" in md  # guardrail note surfaced in the trace


def test_write_trace_creates_file(tmp_path):
    from src.agent import write_trace
    result = AgenticRecommender(_catalog(), k=3).run({"genre": "pop"})
    out = tmp_path / "nested" / "trace.md"
    write_trace([("run", result)], str(out))
    assert out.exists()
    assert "Agent Reasoning Trace Log" in out.read_text()
