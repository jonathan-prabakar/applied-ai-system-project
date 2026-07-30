"""
Evaluation harness for the music recommender agent.

Runs the AgenticRecommender over a set of predefined taste profiles, checks each
result against expectations, and prints a pass/fail summary with a score.

Usage (from the project root):
    python -m src.evaluate
    python -m src.evaluate --verbose     # also show the reason for each check

Exit code is 0 if every scenario passes, 1 otherwise (handy for CI).
"""

import argparse
import sys
from typing import Callable, Dict, List, Tuple

from src.agent import AgenticRecommender, MAX_SAME_GENRE
from src.recommender import load_songs

# Each scenario: (name, user_prefs, k, check_fn).
# check_fn(result) -> (passed: bool, detail: str)
Scenario = Tuple[str, Dict, int, Callable]


def _top_genre_is(genre: str):
    def check(result):
        if not result.recommendations:
            return False, "no recommendations returned"
        top = result.recommendations[0][0]
        ok = top["genre"] == genre
        return ok, f"top pick genre='{top['genre']}', expected '{genre}'"
    return check


def _no_genre_dominates(result):
    genres = [s["genre"] for s, _, _ in result.recommendations]
    worst = max((genres.count(g) for g in set(genres)), default=0)
    ok = worst <= MAX_SAME_GENRE
    return ok, f"max same-genre count={worst}, limit={MAX_SAME_GENRE}"


def _has_warning(substring: str):
    def check(result):
        ok = any(substring in w for w in result.warnings)
        return ok, f"warnings={result.warnings!r}, expected one containing '{substring}'"
    return check


def _returns_k(k: int):
    def check(result):
        ok = len(result.recommendations) == k
        return ok, f"returned {len(result.recommendations)} recs, expected {k}"
    return check


SCENARIOS: List[Scenario] = [
    ("pop/happy -> pop on top", {"genre": "pop", "mood": "happy", "energy": 0.8}, 4,
     _top_genre_is("pop")),
    ("lofi/chill -> diversity enforced", {"genre": "lofi", "mood": "chill", "energy": 0.4}, 5,
     _no_genre_dominates),
    ("out-of-range energy -> clamped warning", {"genre": "pop", "mood": "happy", "energy": 5.0}, 3,
     _has_warning("clamped")),
    ("empty prefs -> handled + returns k", {}, 3,
     _returns_k(3)),
    ("conflict (sad + high energy) -> flagged", {"genre": "rock", "mood": "sad", "energy": 0.9}, 4,
     _has_warning("conflicting")),
]


def run(verbose: bool = False) -> bool:
    songs = load_songs("data/songs.csv")
    width = 68
    print("=" * width)
    print("  MUSIC RECOMMENDER - EVALUATION HARNESS")
    print(f"  {len(SCENARIOS)} scenarios over a {len(songs)}-song catalog")
    print("=" * width)

    passed = 0
    for name, prefs, k, check in SCENARIOS:
        result = AgenticRecommender(songs, k=k).run(prefs)
        ok, detail = check(result)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if verbose or not ok:
            print(f"         -> {detail}")
            print(f"         -> settled in {result.iterations} iteration(s)")

    total = len(SCENARIOS)
    score = 100.0 * passed / total if total else 0.0
    print("-" * width)
    print(f"  SCORE: {passed}/{total} passed  ({score:.0f}%)")
    print("=" * width)
    return passed == total


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommender evaluation harness")
    parser.add_argument("--verbose", action="store_true",
                        help="show the detail line for every scenario")
    args = parser.parse_args()
    all_passed = run(verbose=args.verbose)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
