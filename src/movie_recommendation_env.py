"""
Movie Recommendation Environment - OpenEnv Implementation

A Gymnasium-style environment for training language models to recommend movies
based on user preferences, watch history, and ratings.

Example:
    >>> env = MovieRecommendationEnv()
    >>> result = env.reset(user_profile=user_data)
    >>> result = env.step(RecommendationAction(movie_id="tt1234567"))
    >>> print(result.reward, result.done)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

# Sample movie database
MOVIE_DATABASE = {
    "tt0068646": {"title": "The Godfather", "genres": ["Crime", "Drama"], "year": 1972, "rating": 9.2},
    "tt0071562": {"title": "The Godfather Part II", "genres": ["Crime", "Drama"], "year": 1974, "rating": 9.0},
    "tt0110912": {"title": "Pulp Fiction", "genres": ["Crime", "Drama"], "year": 1994, "rating": 8.9},
    "tt0167260": {"title": "The Lord of the Rings: The Return of the King", "genres": ["Action", "Adventure", "Fantasy"], "year": 2003, "rating": 9.0},
    "tt0108052": {"title": "Schindler's List", "genres": ["Biography", "Drama", "History"], "year": 1993, "rating": 9.0},
    "tt0468569": {"title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"], "year": 2008, "rating": 9.0},
    "tt0050083": {"title": "12 Angry Men", "genres": ["Crime", "Drama"], "year": 1957, "rating": 9.0},
    "tt0816692": {"title": "Interstellar", "genres": ["Adventure", "Drama", "Sci-Fi"], "year": 2014, "rating": 8.7},
    "tt1375666": {"title": "Inception", "genres": ["Action", "Sci-Fi", "Thriller"], "year": 2010, "rating": 8.8},
    "tt0137523": {"title": "Fight Club", "genres": ["Drama"], "year": 1999, "rating": 8.8},
    "tt0944947": {"title": "Game of Thrones", "genres": ["Action", "Adventure", "Drama"], "year": 2011, "rating": 9.2},
    "tt0111161": {"title": "The Shawshank Redemption", "genres": ["Drama"], "year": 1994, "rating": 9.3},
    "tt0073486": {"title": "One Flew Over the Cuckoo's Nest", "genres": ["Drama"], "year": 1975, "rating": 8.7},
    "tt0099685": {"title": "The Stand", "genres": ["Drama", "Horror", "Sci-Fi"], "year": 1994, "rating": 8.4},
    "tt0118715": {"title": "Batman & Robin", "genres": ["Action", "Adventure", "Comedy"], "year": 1997, "rating": 3.7},
}

GENRE_MAPPING = {
    "Drama": "dramatic storytelling",
    "Action": "action-packed sequences",
    "Comedy": "humorous entertainment",
    "Sci-Fi": "science fiction concepts",
    "Horror": "scary content",
    "Adventure": "adventurous narratives",
    "Crime": "crime-related plots",
    "Fantasy": "fantasy elements",
    "Thriller": "suspenseful plots",
    "Biography": "biographical stories",
    "History": "historical events",
}


class MessageCategory(str, Enum):
    """Message category types in the environment."""
    PROMPT = "PROMPT"
    USER_PROFILE = "USER_PROFILE"
    WATCH_HISTORY = "WATCH_HISTORY"
    RECOMMENDATION = "RECOMMENDATION"
    FEEDBACK = "FEEDBACK"
    ERROR = "ERROR"


@dataclass
class Message:
    """A single message in the conversation history."""
    category: str
    content: str


@dataclass
class Observation:
    """Observation returned by the environment."""
    prompt: str
    messages: list[Message]
    context: str


@dataclass
class StepResult:
    """Result from a step in the environment."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class ResetResult:
    """Result from resetting the environment."""
    observation: Observation
    info: dict[str, Any]


@dataclass
class RecommendationAction:
    """Action: recommend a movie by ID or title."""
    movie_id: Optional[str] = None
    movie_title: Optional[str] = None


class MovieRecommendationEnv:
    """
    OpenEnv-compatible Movie Recommendation Environment.
    
    The agent must recommend movies based on a user's profile, preferences,
    and watch history. Rewards are based on:
    - Relevance to user genre preferences
    - Quality of the recommendation
    - Rating compatibility with user's typical ratings
    """

    def __init__(self, max_recommendations: int = 6, movie_db: Optional[dict] = None):
        """
        Initialize the environment.
        
        Args:
            max_recommendations: Maximum recommendations per episode
            movie_db: Custom movie database (defaults to MOVIE_DATABASE)
        """
        self.max_recommendations = max_recommendations
        self.movie_db = movie_db or MOVIE_DATABASE
        self.current_step = 0
        self.user_profile = None
        self.watch_history = []
        self.recommendations = []
        self.messages: list[Message] = []
        self.done = False

    def reset(
        self,
        user_profile: Optional[dict] = None,
        watch_history: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> ResetResult:
        """
        Reset the environment with a user profile.
        
        Args:
            user_profile: Dict with 'genres', 'avg_rating_tolerance'
            watch_history: List of movie IDs the user has watched
            seed: Random seed for reproducibility
            
        Returns:
            ResetResult with initial observation
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Default user profile
        if user_profile is None:
            user_profile = {
                "name": f"User_{random.randint(1000, 9999)}",
                "preferred_genres": random.sample(list(GENRE_MAPPING.keys()), k=2),
                "min_rating": round(random.uniform(6.0, 8.0), 1),
                "avg_rating_tolerance": round(random.uniform(0.5, 2.0), 1),
            }

        if watch_history is None:
            watch_history = random.sample(list(self.movie_db.keys()), k=random.randint(2, 5))

        self.user_profile = user_profile
        self.watch_history = watch_history
        self.recommendations = []
        self.messages = []
        self.done = False
        self.current_step = 0

        # Build initial observation
        profile_text = self._format_user_profile(user_profile, watch_history)
        initial_prompt = (
            f"You are an expert movie recommendation agent. Based on the user profile below, "
            f"recommend movies they would enjoy.\n\n{profile_text}"
        )

        self.messages.append(Message(
            category=MessageCategory.PROMPT,
            content="Recommend a movie for this user."
        ))
        self.messages.append(Message(
            category=MessageCategory.USER_PROFILE,
            content=profile_text
        ))

        observation = Observation(
            prompt=initial_prompt,
            messages=self.messages.copy(),
            context=profile_text,
        )

        return ResetResult(observation=observation, info={"user_id": user_profile.get("name")})

    def step(self, action: RecommendationAction) -> StepResult:
        """
        Process a recommendation action.
        
        Args:
            action: RecommendationAction with movie_id or movie_title
            
        Returns:
            StepResult with observation, reward, done flag, and info
        """
        self.current_step += 1
        info = {}

        # Find the movie
        movie_id = action.movie_id
        if movie_id is None and action.movie_title:
            movie_id = self._find_movie_by_title(action.movie_title)

        if movie_id is None:
            reward = -1.0
            info["error"] = "Movie not found"
            self.messages.append(Message(
                category=MessageCategory.ERROR,
                content=f"Could not find movie: {action.movie_id or action.movie_title}"
            ))
        elif movie_id in self.watch_history or movie_id in self.recommendations:
            reward = -0.5
            info["error"] = "Movie already in watch history or recommended"
            self.messages.append(Message(
                category=MessageCategory.ERROR,
                content=f"Movie {self._get_movie_title(movie_id)} already watched or recommended"
            ))
        else:
            # Calculate reward based on relevance
            reward = self._calculate_reward(movie_id)
            self.recommendations.append(movie_id)
            movie = self.movie_db[movie_id]

            self.messages.append(Message(
                category=MessageCategory.RECOMMENDATION,
                content=f"Recommended: {movie['title']} (Rating: {movie['rating']}, Genres: {', '.join(movie['genres'])})"
            ))

            feedback = self._generate_feedback(movie_id, reward)
            self.messages.append(Message(
                category=MessageCategory.FEEDBACK,
                content=feedback
            ))

            info["movie_id"] = movie_id
            info["movie_title"] = movie["title"]
            info["reward_breakdown"] = self._reward_breakdown(movie_id)

        # Check if episode is done
        done = self.current_step >= self.max_recommendations
        self.done = done

        observation = Observation(
            prompt=f"Continue recommending movies. Recommendations so far: {len(self.recommendations)}/{self.max_recommendations}",
            messages=self.messages.copy(),
            context=self._format_user_profile(self.user_profile, self.watch_history),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def _format_user_profile(self, profile: dict, watch_history: list) -> str:
        """Format user profile and watch history as readable text."""
        profile_text = f"User: {profile['name']}\n"
        profile_text += f"Preferred Genres: {', '.join(profile['preferred_genres'])}\n"
        profile_text += f"Minimum Rating Threshold: {profile['min_rating']}\n"
        profile_text += f"Rating Tolerance: ±{profile['avg_rating_tolerance']}\n"

        watched_count = min(3, len(watch_history))
        profile_text += f"\nRecent Watch History ({watched_count} of {len(watch_history)}):\n"
        for movie_id in watch_history[:watched_count]:
            movie = self.movie_db.get(movie_id)
            if movie:
                profile_text += f"  - {movie['title']} (Rating: {movie['rating']})\n"

        return profile_text

    def _find_movie_by_title(self, title: str) -> Optional[str]:
        """Find movie ID by title (case-insensitive, partial match)."""
        title_lower = title.lower()
        for movie_id, movie in self.movie_db.items():
            if title_lower in movie["title"].lower() or movie["title"].lower() in title_lower:
                return movie_id
        return None

    def _get_movie_title(self, movie_id: str) -> str:
        """Get movie title from ID."""
        return self.movie_db.get(movie_id, {}).get("title", "Unknown")

    def _calculate_reward(self, movie_id: str) -> float:
        """
        Calculate reward for a recommendation.
        
        Reward factors:
        - Genre match (0 to 0.5)
        - Rating quality (0 to 0.3)
        - Popularity/Recency bonus (0 to 0.2)
        """
        if movie_id not in self.movie_db:
            return -1.0

        movie = self.movie_db[movie_id]
        profile = self.user_profile
        reward = 0.0

        # Genre matching reward (0 to 0.5)
        genre_overlap = len(set(movie["genres"]) & set(profile["preferred_genres"]))
        genre_reward = min(0.5, (genre_overlap / len(profile["preferred_genres"])) * 0.5)
        reward += genre_reward

        # Rating quality reward (0 to 0.3)
        rating_diff = abs(movie["rating"] - profile["min_rating"])
        if rating_diff <= profile["avg_rating_tolerance"]:
            rating_reward = 0.3 * (1 - (rating_diff / (profile["avg_rating_tolerance"] + 0.1)))
        else:
            rating_reward = 0.1 * (1 - min(1.0, rating_diff / 5.0))
        reward += rating_reward

        # High quality bonus (0 to 0.2)
        if movie["rating"] >= 8.5:
            reward += 0.2
        elif movie["rating"] >= 8.0:
            reward += 0.1

        return min(1.0, reward)

    def _reward_breakdown(self, movie_id: str) -> dict:
        """Return breakdown of reward components."""
        return {
            "genre_match": self._genre_match_score(movie_id),
            "rating_quality": self._rating_quality_score(movie_id),
            "overall": self._calculate_reward(movie_id),
        }

    def _genre_match_score(self, movie_id: str) -> float:
        """Calculate genre matching score."""
        movie = self.movie_db[movie_id]
        profile = self.user_profile
        genre_overlap = len(set(movie["genres"]) & set(profile["preferred_genres"]))
        return min(1.0, (genre_overlap / len(profile["preferred_genres"])))

    def _rating_quality_score(self, movie_id: str) -> float:
        """Calculate rating quality score."""
        movie = self.movie_db[movie_id]
        return min(1.0, movie["rating"] / 10.0)

    def _generate_feedback(self, movie_id: str, reward: float) -> str:
        """Generate human-readable feedback for the recommendation."""
        movie = self.movie_db[movie_id]
        profile = self.user_profile

        if reward > 0.7:
            feedback_base = f"✅ Excellent recommendation! {movie['title']} aligns well with"
        elif reward > 0.5:
            feedback_base = f"👍 Good recommendation. {movie['title']} matches some of"
        elif reward > 0.2:
            feedback_base = f"⚠️  {movie['title']} might interest"
        else:
            feedback_base = f"❌ {movie['title']} doesn't match"

        genres_desc = ", ".join([GENRE_MAPPING.get(g, g) for g in movie['genres']])
        feedback = f"{feedback_base} the user's preferences ({genres_desc})."

        return feedback

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current state."""
        if mode == "human":
            print("\n" + "=" * 60)
            print(f"Step {self.current_step}/{self.max_recommendations}")
            print("Messages:")
            for msg in self.messages:
                print(f"  [{msg.category}] {msg.content}")
            print("=" * 60)
        return None

    def close(self):
        """Clean up environment resources."""
        self.messages = []
        self.recommendations = []

    def get_state(self) -> dict:
        """Get current environment state."""
        return {
            "user_profile": self.user_profile,
            "watch_history": self.watch_history,
            "recommendations": self.recommendations,
            "current_step": self.current_step,
            "done": self.done,
            "messages": [asdict(m) for m in self.messages],
        }

    def __repr__(self) -> str:
        return (
            f"MovieRecommendationEnv("
            f"steps={self.current_step}/{self.max_recommendations}, "
            f"recommendations={len(self.recommendations)})"
        )
