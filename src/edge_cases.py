"""
Run the recommender against adversarial / edge-case taste profiles.

Usage (from the project root):
    python -m src.edge_cases
"""

from src.recommender import load_songs, recommend_songs


# Each entry: (label, user_prefs). Add your own profiles here.
EDGE_CASES = [
    ("Conflicting: high energy + sad mood", {"genre": "pop", "mood": "sad", "energy": 0.9}),
    ("Out-of-range energy (5.0)",           {"genre": "pop", "mood": "happy", "energy": 5.0}),
    ("Empty preferences",                   {}),
    ("Nonsense genre/mood, no acoustic",    {"genre": "zzz", "mood": "zzz", "energy": 0.5, "likes_acoustic": False}),
    ("Near-miss genre ('hiphop')",          {"genre": "hiphop", "mood": "confident", "energy": 0.78}),
]


def main() -> None:
    songs = load_songs("data/songs.csv")
    width = 70

    for label, prefs in EDGE_CASES:
        print("=" * width)
        print(f"  {label}")
        print(f"  profile: {prefs}")
        print("=" * width)

        recommendations = recommend_songs(prefs, songs, k=3)
        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"\n  {rank}. {song['title']} — {song['artist']}  (score {score:.2f})")
            print(f"     {song['genre']} / {song['mood']} / energy {song['energy']}")
            for reason in explanation.split(", "):
                print(f"       • {reason}")
        print()


if __name__ == "__main__":
    main()
