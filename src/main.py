"""
Command line runner for the Music Recommender Simulation.

The recommender now runs as an AGENTIC WORKFLOW: instead of scoring the catalog
once, it PLANs a strategy, ACTs to produce recommendations, CHECKs its own
output against quality guardrails, and FIXes the strategy until the checks pass
(or a safe iteration cap is hit). See src/agent.py for the loop.

Run:
    python -m src.main                 # normal run
    python -m src.main --verbose       # also stream the agent's internal logs
"""

import argparse
import logging

from src.agent import AgenticRecommender, write_trace
from src.recommender import load_songs


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic music recommender")
    parser.add_argument("--verbose", action="store_true",
                        help="stream the agent's PLAN/ACT/CHECK/FIX logs")
    parser.add_argument("--data", default="data/songs.csv", help="path to song CSV")
    parser.add_argument("-k", type=int, default=5, help="how many songs to recommend")
    parser.add_argument("--trace-out", metavar="PATH",
                        help="write the agent's full reasoning chain to a Markdown log")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    try:
        songs = load_songs(args.data)
    except FileNotFoundError:
        raise SystemExit(f"Could not find song data at '{args.data}'. "
                         "Run from the project root or pass --data.")
    if not songs:
        raise SystemExit(f"No songs loaded from '{args.data}'.")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    agent = AgenticRecommender(songs, k=args.k)
    result = agent.run(user_prefs)

    profile = f"{user_prefs['genre']} / {user_prefs['mood']} / energy {user_prefs['energy']}"
    width = 60

    # Show the agent's reasoning trace so the workflow is visible, not hidden.
    print("\n" + "-" * width)
    print("  AGENT WORKFLOW TRACE (plan -> act -> check -> fix)")
    print("-" * width)
    for step in result.trace:
        print(f"  {step}")
    print(f"  -> settled after {result.iterations} iteration(s)")

    print("\n" + "=" * width)
    print("  TOP RECOMMENDATIONS".ljust(width))
    print(f"  for taste profile: {profile}".ljust(width))
    print("=" * width)

    for rank, (song, score, explanation) in enumerate(result.recommendations, start=1):
        title = f"{song['title']} — {song['artist']}"
        print(f"\n  {rank}. {title}")
        print(f"     Score: {score:.2f}")
        print("     Reasons:")
        for reason in explanation.split(", "):
            print(f"       • {reason}")

    if result.warnings:
        print("\n" + "-" * width)
        print("  GUARDRAIL NOTES")
        print("-" * width)
        for warning in result.warnings:
            print(f"  ! {warning}")

    print("\n" + "=" * width)

    if args.trace_out:
        write_trace([(f"Request: {profile}", result)], args.trace_out)
        print(f"\nFull reasoning chain written to: {args.trace_out}")


if __name__ == "__main__":
    main()
