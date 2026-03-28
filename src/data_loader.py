"""
Data Loading Utilities for Movie Recommendation Environment

Supports:
- Built-in demo database (15 movies)
- MovieLens 100K dataset (1682 movies, 100K ratings)
- MovieLens 1M dataset (3883 movies, 1M ratings)
- Custom movie databases (JSON/CSV)
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
import io


class MovieDatabase:
    """Movie database loader and converter."""

    @staticmethod
    def load_builtin() -> Dict[str, Dict]:
        """Load the built-in demo movie database."""
        from movie_recommendation_env import MOVIE_DATABASE
        return MOVIE_DATABASE.copy()

    @staticmethod
    def load_json(filepath: str) -> Dict[str, Dict]:
        """Load movie database from JSON file.
        
        Expected format:
        {
            "tt0111161": {
                "title": "The Shawshank Redemption",
                "genres": ["Drama"],
                "year": 1994,
                "rating": 9.3
            },
            ...
        }
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(movies: Dict[str, Dict], filepath: str) -> None:
        """Save movie database to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(movies, f, indent=2)

    @staticmethod
    def from_movielens_100k(data_dir: str = "./data/movielens/ml-100k") -> Dict[str, Dict]:
        """
        Convert MovieLens 100K dataset to our format.
        
        Args:
            data_dir: Path to MovieLens 100K extracted directory
            
        Returns:
            Dictionary in our movie database format
        """
        movies_filepath = Path(data_dir) / "u.item"
        genre_filepath = Path(data_dir) / "u.genre"
        rating_filepath = Path(data_dir) / "u.data"

        if not movies_filepath.exists():
            raise FileNotFoundError(
                f"MovieLens 100K data not found at {data_dir}. "
                "Run: python scripts/download_movielens.py"
            )

        # Load genres mapping
        genres_map = {}
        if genre_filepath.exists():
            with open(genre_filepath, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('|')
                        if len(parts) >= 2:
                            genres_map[int(parts[1])] = parts[0]

        # Load movies and ratings
        movies = {}
        ratings_by_id = {}  # For averaging ratings

        with open(rating_filepath, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    movie_id = int(parts[1])
                    rating = float(parts[2])
                    if movie_id not in ratings_by_id:
                        ratings_by_id[movie_id] = []
                    ratings_by_id[movie_id].append(rating)

        with open(movies_filepath, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 6:
                    movie_id = int(parts[0])
                    title = parts[1]
                    year_str = parts[2]
                    year = int(year_str[-4:]) if year_str and len(year_str) >= 4 else 2000

                    # Extract genres
                    genres = []
                    genre_indices = parts[5:]
                    for idx, has_genre in enumerate(genre_indices):
                        if has_genre == '1' and idx in genres_map:
                            genres.append(genres_map[idx])

                    if not genres:
                        genres = ["Unknown"]

                    # Calculate average rating (normalized to 10)
                    if movie_id in ratings_by_id:
                        ratings = ratings_by_id[movie_id]
                        avg_rating = sum(ratings) / len(ratings) * 2  # MovieLens is 1-5, convert to 1-10
                        avg_rating = min(10.0, max(1.0, avg_rating))  # Clamp to 1-10
                    else:
                        avg_rating = 5.0

                    movies[f"ml_{movie_id:05d}"] = {
                        "title": title,
                        "genres": genres[:3],  # Limit to 3 genres
                        "year": year,
                        "rating": round(avg_rating, 1),
                    }

        print(f"✅ Loaded {len(movies)} movies from MovieLens 100K")
        return movies

    @staticmethod
    def from_movielens_1m(data_dir: str = "./data/movielens/ml-1m") -> Dict[str, Dict]:
        """
        Convert MovieLens 1M dataset to our format.
        
        Args:
            data_dir: Path to MovieLens 1M extracted directory
            
        Returns:
            Dictionary in our movie database format
        """
        movies_filepath = Path(data_dir) / "movies.dat"
        rating_filepath = Path(data_dir) / "ratings.dat"

        if not movies_filepath.exists():
            raise FileNotFoundError(
                f"MovieLens 1M data not found at {data_dir}. "
                "Run: python scripts/download_movielens.py --version 1m"
            )

        # Load ratings
        ratings_by_id = {}
        with open(rating_filepath, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = int(parts[1])
                    rating = float(parts[2])
                    if movie_id not in ratings_by_id:
                        ratings_by_id[movie_id] = []
                    ratings_by_id[movie_id].append(rating)

        # Load movies
        movies = {}
        with open(movies_filepath, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = int(parts[0])
                    title = parts[1]
                    genres_str = parts[2]

                    # Extract year from title
                    year = 2000
                    if '(' in title and ')' in title:
                        year_part = title.split('(')[-1].split(')')[0]
                        try:
                            year = int(year_part)
                        except ValueError:
                            pass

                    # Extract genres
                    genres = [g.strip() for g in genres_str.split('|')][:3]
                    if not genres or genres == ['']:
                        genres = ["Unknown"]

                    # Calculate average rating (normalize to 1-10)
                    if movie_id in ratings_by_id:
                        ratings = ratings_by_id[movie_id]
                        avg_rating = sum(ratings) / len(ratings) * 2
                        avg_rating = min(10.0, max(1.0, avg_rating))
                    else:
                        avg_rating = 5.0

                    movies[f"ml_{movie_id:05d}"] = {
                        "title": title.split('(')[0].strip(),
                        "genres": genres,
                        "year": year,
                        "rating": round(avg_rating, 1),
                    }

        print(f"✅ Loaded {len(movies)} movies from MovieLens 1M")
        return movies

    @staticmethod
    def merge_databases(databases: List[Dict[str, Dict]]) -> Dict[str, Dict]:
        """Merge multiple movie databases, avoiding duplicates."""
        merged = {}
        for db in databases:
            for movie_id, movie_info in db.items():
                if movie_id not in merged:
                    merged[movie_id] = movie_info
        return merged

    @staticmethod
    def sample_movies(
        movies: Dict[str, Dict],
        num_samples: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, Dict]:
        """Sample subset of movies from database."""
        import random
        if seed is not None:
            random.seed(seed)
        
        movie_ids = list(movies.keys())
        if len(movie_ids) <= num_samples:
            return movies.copy()
        
        sampled_ids = random.sample(movie_ids, num_samples)
        return {mid: movies[mid] for mid in sampled_ids}

    @staticmethod
    def filter_by_rating(
        movies: Dict[str, Dict],
        min_rating: float = 5.0,
        max_rating: float = 10.0
    ) -> Dict[str, Dict]:
        """Filter movies by rating range."""
        return {
            mid: info for mid, info in movies.items()
            if min_rating <= info.get('rating', 5.0) <= max_rating
        }

    @staticmethod
    def filter_by_genre(
        movies: Dict[str, Dict],
        genres: List[str]
    ) -> Dict[str, Dict]:
        """Filter movies that contain any of the specified genres."""
        filtered = {}
        for mid, info in movies.items():
            movie_genres = info.get('genres', [])
            if any(g in movie_genres for g in genres):
                filtered[mid] = info
        return filtered


class MovieLensDownloader:
    """Download and extract MovieLens datasets."""

    ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    @staticmethod
    def download_100k(output_dir: str = "./data/movielens") -> bool:
        """
        Download MovieLens 100K dataset.
        
        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading MovieLens 100K dataset...")
        print(f"   Source: {MovieLensDownloader.ML_100K_URL}")

        try:
            response = requests.get(MovieLensDownloader.ML_100K_URL, stream=True)
            response.raise_for_status()

            # Extract zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(output_path)

            print(f"✅ Downloaded and extracted to {output_path}/ml-100k")
            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    @staticmethod
    def download_1m(output_dir: str = "./data/movielens") -> bool:
        """
        Download MovieLens 1M dataset.
        
        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading MovieLens 1M dataset (this may take a few minutes)...")
        print(f"   Source: {MovieLensDownloader.ML_1M_URL}")

        try:
            response = requests.get(MovieLensDownloader.ML_1M_URL, stream=True, timeout=60)
            response.raise_for_status()

            # Extract zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(output_path)

            print(f"✅ Downloaded and extracted to {output_path}/ml-1m")
            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
