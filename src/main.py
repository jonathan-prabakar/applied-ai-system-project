"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    profile = f"{user_prefs['genre']} / {user_prefs['mood']} / energy {user_prefs['energy']}"
    width = 60
    print()
    print("=" * width)
    print("  TOP RECOMMENDATIONS".ljust(width))
    print(f"  for taste profile: {profile}".ljust(width))
    print("=" * width)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        title = f"{song['title']} — {song['artist']}"
        print(f"\n  {rank}. {title}")
        print(f"     Score: {score:.2f}")
        print("     Reasons:")
        for reason in explanation.split(", "):
            print(f"       • {reason}")

    print("\n" + "=" * width)


if __name__ == "__main__":
    main()
