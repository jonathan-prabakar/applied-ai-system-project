import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# --- Scoring weights -----------------------------------------------------
# Categorical matches are worth flat points; numeric features score by how
# CLOSE the song is to the user's preference (1.0 = perfect, 0.0 = opposite).
W_GENRE = 2.0
W_MOOD = 3.0          # mood best captures "vibe", so it outweighs genre
W_ENERGY = 2.0
W_DANCEABILITY = 1.5  # rewards songs whose groove matches the user's taste
W_ACOUSTIC = 1.0


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_danceability: float = 0.5  # how groovy the user likes it (0-1)


def _closeness(value: float, target: float) -> float:
    """Reward being close to the target: 1.0 = exact match, 0.0 = opposite.

    Both value and target must be on the same 0-1 scale.
    """
    return 1.0 - abs(value - target)


def score_song(user: UserProfile, song: Song) -> Tuple[float, List[str]]:
    """Score a song against a user's taste.

    Returns (score, reasons) where `reasons` explains each point awarded, e.g.
    ["genre match (+2.0)", "energy close (+1.96)"], so the user can see WHY a
    song was recommended.

    Point rules:
      - genre match:  flat +2.0 (exact category match)
      - mood match:   flat +3.0 (mood best captures "vibe", so worth the most)
      - energy:       up to +2.0, scaled by how CLOSE the song is to the user's
                      target -> W_ENERGY * (1 - |song.energy - target_energy|)
      - danceability: up to +1.5, same closeness idea
      - acoustic:     flat +1.0 when the song's acoustic-ness matches the user
    """
    score = 0.0
    reasons: List[str] = []

    if song.genre == user.favorite_genre:
        score += W_GENRE
        reasons.append(f"genre match ({song.genre}) (+{W_GENRE:.1f})")
    if song.mood == user.favorite_mood:
        score += W_MOOD
        reasons.append(f"mood match ({song.mood}) (+{W_MOOD:.1f})")

    energy_pts = W_ENERGY * _closeness(song.energy, user.target_energy)
    score += energy_pts
    reasons.append(f"energy close (+{energy_pts:.2f})")

    dance_pts = W_DANCEABILITY * _closeness(song.danceability, user.target_danceability)
    score += dance_pts
    reasons.append(f"danceability close (+{dance_pts:.2f})")

    # acousticness > 0.6 is treated as an "acoustic" song
    if (song.acousticness > 0.6) == user.likes_acoustic:
        score += W_ACOUSTIC
        reasons.append(f"acoustic preference match (+{W_ACOUSTIC:.1f})")

    return score, reasons


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked highest-first by their taste score."""
        ranked = sorted(self.songs, key=lambda s: score_song(user, s)[0], reverse=True)
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a one-line sentence explaining why a song scored as it did."""
        score, reasons = score_song(user, song)
        return (
            f"'{song.title}' by {song.artist} scored {score:.2f}: "
            + ", ".join(reasons)
            + "."
        )


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    numeric = {"id", "energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in numeric:
                if key in row:
                    row[key] = int(row[key]) if key == "id" else float(row[key])
            songs.append(row)
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Returns a list of (song_dict, score, explanation) sorted by score.
    """
    # Score every song, then sort highest-first and keep the top k.
    scored = [_score_song_dict(user_prefs, song) for song in songs]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def _score_song_dict(user_prefs: Dict, song: Dict) -> Tuple[Dict, float, str]:
    """Score one song dict; mirrors score_song() for the functional API."""
    target_energy = user_prefs.get("energy", 0.5)
    target_dance = user_prefs.get("danceability", 0.5)
    likes_acoustic = user_prefs.get("likes_acoustic", False)

    score = 0.0
    reasons: List[str] = []
    if song["genre"] == user_prefs.get("genre"):
        score += W_GENRE
        reasons.append(f"genre match ({song['genre']}) (+{W_GENRE:.1f})")
    if song["mood"] == user_prefs.get("mood"):
        score += W_MOOD
        reasons.append(f"mood match ({song['mood']}) (+{W_MOOD:.1f})")

    energy_pts = W_ENERGY * _closeness(song["energy"], target_energy)
    score += energy_pts
    reasons.append(f"energy close (+{energy_pts:.2f})")

    dance_pts = W_DANCEABILITY * _closeness(song["danceability"], target_dance)
    score += dance_pts
    reasons.append(f"danceability close (+{dance_pts:.2f})")

    if (song["acousticness"] > 0.6) == likes_acoustic:
        score += W_ACOUSTIC
        reasons.append(f"acoustic preference match (+{W_ACOUSTIC:.1f})")

    return song, score, ", ".join(reasons)
