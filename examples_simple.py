"""
Simple Example: Test the Movie Recommendation Environment

Shows how to interact with the environment without any LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recommendation_env import (
    MovieRecommendationEnv,
    RecommendationAction,
    GENRE_MAPPING,
)


def main():
    """Run a simple recommendation episode."""
    print("=" * 70)
    print("🎬 SIMPLE MOVIE RECOMMENDATION ENVIRONMENT DEMO")
    print("=" * 70)
    print()

    # Create environment
    env = MovieRecommendationEnv(max_recommendations=6)

    # Create a user profile
    user_profile = {
        "name": "Alice",
        "preferred_genres": ["Drama", "Crime"],
        "min_rating": 8.0,
        "avg_rating_tolerance": 1.5,
    }

    # Watch history
    watch_history = ["tt0068646", "tt0110912"]  # The Godfather, Pulp Fiction

    # Reset environment
    print(f"User: {user_profile['name']}")
    print(f"Preferred Genres: {', '.join(user_profile['preferred_genres'])}")
    print(f"Minimum Rating: {user_profile['min_rating']}")
    print(f"Watch History: {len(watch_history)} movies\n")

    result = env.reset(user_profile=user_profile, watch_history=watch_history)

    print("Initial Observation:")
    print(f"Prompt: {result.observation.prompt[:100]}...\n")

    # Make recommendations manually
    recommendations = [
        "tt0111161",  # The Shawshank Redemption
        "tt0108052",  # Schindler's List
        "tt0050083",  # 12 Angry Men
        "tt0816692",  # Interstellar (different genre)
        "tt0468569",  # The Dark Knight
        "tt1375666",  # Inception
    ]

    total_reward = 0
    successful_recs = 0

    for step, movie_id in enumerate(recommendations, 1):
        print(f"\n{'=' * 70}")
        print(f"Step {step}/6 - Recommending:")
        print(f"{'=' * 70}")

        action = RecommendationAction(movie_id=movie_id)
        result = env.step(action)

        reward = result.reward
        total_reward += reward
        done = result.done

        # Show feedback
        if "error" in result.info:
            print(f"❌ Error: {result.info['error']}")
        else:
            print(f"✅ Success!")
            successful_recs += 1
            movie_title = result.info.get("movie_title", "Unknown")
            print(f"   Title: {movie_title}")
            print(f"   Reward: {reward:.3f}")

            if "reward_breakdown" in result.info:
                breakdown = result.info["reward_breakdown"]
                print(f"   Genre Match: {breakdown.get('genre_match', 0):.3f}")
                print(f"   Quality Score: {breakdown.get('rating_quality', 0):.3f}")

        # Show last message
        if result.observation.messages:
            last_msg = result.observation.messages[-1]
            print(f"   Feedback: {last_msg.content}")

        print(f"   Done: {done}")

        if done:
            print("\n🏁 Episode completed!")
            break

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 EPISODE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total Steps: {step}")
    print(f"Successful Recommendations: {successful_recs}/{step}")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Average Reward: {total_reward / step:.3f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
