"""
Inference Script for Movie Recommendation Agent

Use a trained model to get movie recommendations for a user profile.

Usage:
    python inference.py --model-id path/to/model --user-genres Drama,Crime
    
    # With device selection
    python inference.py --device cuda    # NVIDIA GPU
    python inference.py --device mps     # Apple Silicon Mac
    python inference.py --device cpu     # CPU only
    python inference.py               # Auto-detect best device
    
Device Support:
    - CUDA: NVIDIA GPUs (recommended)
    - MPS: Apple Silicon Macs (auto-converted to CPU-loaded then moved to MPS)
    - CPU: Works everywhere (slower)
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recommendation_env import (
    MovieRecommendationEnv,
    RecommendationAction,
    GENRE_MAPPING,
)
from data_loader import MovieDatabase


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained movie recommendation model"
    )
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Fine-tuned model identifier or path",
    )
    parser.add_argument(
        "--user-genres",
        default="Drama,Action",
        help="Comma-separated user genre preferences",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=7.5,
        help="Minimum movie rating threshold",
    )
    parser.add_argument(
        "--num-recommendations",
        type=int,
        default=6,
        help="Number of recommendations to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Device to run on (auto/cuda/cpu/mps). Default: auto-detect",
    )
    parser.add_argument(
        "--movie-database",
        default="movielens_100k",
        help="Movie database to use (default: movielens_100k). "
             "Can be 'movielens_100k' (1,682 movies), 'movielens_1m' (3,883 movies), "
             "or path to custom JSON file.",
    )

    return parser.parse_args()


def load_movie_database(db_arg):
    """Load movie database from various sources.
    
    Args:
        db_arg: None (built-in), file path, or 'movielens_100k'/'movielens_1m'
        
    Returns:
        Movie database dictionary
    """
    from movie_recommendation_env import MOVIE_DATABASE
    
    if db_arg is None:
        print("📚 Using built-in movie database (15 movies)")
        return MOVIE_DATABASE.copy()
    
    if db_arg == "movielens_100k":
        print("📚 Loading MovieLens 100K dataset")
        return MovieDatabase.from_movielens_100k()
    
    if db_arg == "movielens_1m":
        print("📚 Loading MovieLens 1M dataset")
        return MovieDatabase.from_movielens_1m()
    
    # Try loading from file path
    filepath = Path(db_arg)
    if filepath.exists():
        print(f"📚 Loading movie database from {filepath}")
        return MovieDatabase.load_json(str(filepath))
    
    raise FileNotFoundError(
        f"Movie database not found: {db_arg}. "
        "Please provide valid path or use 'movielens_100k'/'movielens_1m'."
    )


def get_device(device_arg: str) -> str:
    """Get appropriate device based on availability and user preference."""
    if device_arg != "auto":
        if device_arg == "mps" and not torch.backends.mps.is_available():
            print("⚠️  MPS device requested but not available, falling back to CPU")
            return "cpu"
        return device_arg
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 CUDA device detected")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Metal Performance Shaders (MPS) device detected")
    else:
        device = "cpu"
        print("💻 CPU mode (no GPU acceleration available)")
    
    return device


def get_model_and_tokenizer(model_id: str, device: str):
    """Load model and tokenizer with proper device handling."""
    print(f"Loading model from {model_id}...")
    print(f"Device: {device}")
    
    # For MPS, use device_map="cpu" and then move to MPS after loading
    # This avoids issues with some operations not yet supported on MPS
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load on CPU first for MPS compatibility
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else device,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"✅ Model loaded on device: {device}\n")
    return model, tokenizer


def generate_recommendation(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    max_tokens: int = 32,
):
    """Generate a single movie recommendation."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()


def extract_movie_from_response(response: str) -> Optional[str]:
    """Extract movie recommendation from model response."""
    import re
    match = re.search(r"\[([^\]]+)\]", response)
    if match:
        return match.group(1).strip()
    return None


def main():
    """Main inference loop."""
    args = parse_args()
    
    # Get appropriate device
    device = get_device(args.device)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_id, device)
    
    # Load system prompt
    system_prompt_path = Path(__file__).parent / "src" / "system_prompt.txt"
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text()
    else:
        system_prompt = "You are an expert movie recommendation agent."
    
    # Parse user preferences
    genres = [g.strip() for g in args.user_genres.split(",")]
    genres = [g for g in genres if g in GENRE_MAPPING]
    
    if not genres:
        print(f"❌ Invalid genres. Available genres: {', '.join(GENRE_MAPPING.keys())}")
        return
    
    # Create user profile
    user_profile = {
        "name": "InferenceUser",
        "preferred_genres": genres,
        "min_rating": args.min_rating,
        "avg_rating_tolerance": 1.5,
    }
    
    # Load movie database
    movie_db = load_movie_database(args.movie_database)
    print(f"Loaded {len(movie_db)} movies for recommendations\n")
    
    # Initialize environment
    env = MovieRecommendationEnv(
        max_recommendations=args.num_recommendations,
        movie_db=movie_db
    )
    result = env.reset(user_profile=user_profile)
    observation = result.observation
    
    print("\n" + "=" * 70)
    print("🎬 MOVIE RECOMMENDATION INFERENCE")
    print("=" * 70)
    print(f"User Profile: {user_profile['name']}")
    print(f"Preferred Genres: {', '.join(genres)}")
    print(f"Minimum Rating: {args.min_rating}")
    print(f"Recommendations to generate: {args.num_recommendations}")
    print("=" * 70 + "\n")
    
    recommendations = []
    
    for step in range(args.num_recommendations):
        if observation.done:
            break
        
        print(f"Step {step + 1}/{args.num_recommendations}")
        print("-" * 40)
        
        # Build user prompt
        from movie_recommendation_env import Message
        history_lines = []
        for msg in observation.messages:
            if msg.category in ["RECOMMENDATION", "FEEDBACK"]:
                history_lines.append(f"[{msg.category}] {msg.content}")
        
        history = "\n".join(history_lines[-3:]) if history_lines else "No history yet"
        
        user_prompt = (
            f"Genres to recommend: {', '.join(genres)}\n"
            f"User's minimum rating: {args.min_rating}\n"
            f"Recent feedback:\n{history}\n\n"
            f"Recommend the next movie:"
        )
        
        # Generate recommendation
        print("Generating recommendation...")
        response = generate_recommendation(
            model,
            tokenizer,
            system_prompt,
            user_prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print(f"Model response: {response}")
        
        # Extract movie
        movie_title = extract_movie_from_response(response)
        if not movie_title:
            print("⚠️  No valid movie format detected, using fallback")
            movie_title = "The Shawshank Redemption"
        
        # Step environment
        action = RecommendationAction(movie_title=movie_title)
        result = env.step(action)
        observation = result.observation
        
        # Show result
        reward = result.reward
        info = result.info
        
        if "error" in info:
            print(f"❌ {info['error']}")
        else:
            movie_id = info.get("movie_id", "unknown")
            title = info.get("movie_title", movie_title)
            print(f"✅ Recommended: {title}")
            print(f"   Reward Score: {reward:.3f}")
            if "reward_breakdown" in info:
                breakdown = info["reward_breakdown"]
                print(f"   Genre Match: {breakdown.get('genre_match', 0):.2f}")
                print(f"   Quality Score: {breakdown.get('rating_quality', 0):.2f}")
            recommendations.append({
                "step": step + 1,
                "title": title,
                "movie_id": movie_id,
                "reward": reward,
            })
        
        print()
    
    # Print summary
    print("=" * 70)
    print("📊 RECOMMENDATION SUMMARY")
    print("=" * 70)
    for rec in recommendations:
        print(f"{rec['step']}. {rec['title']:<40} (Reward: {rec['reward']:.3f})")
    
    total_reward = sum(rec['reward'] for rec in recommendations)
    avg_reward = total_reward / len(recommendations) if recommendations else 0
    
    print(f"\nTotal Reward: {total_reward:.3f}")
    print(f"Average Reward per Recommendation: {avg_reward:.3f}")
    print("=" * 70)
    
    # Save results
    output_file = Path("recommendation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "user_profile": user_profile,
            "recommendations": recommendations,
            "summary": {
                "total_reward": total_reward,
                "average_reward": avg_reward,
                "num_recommendations": len(recommendations),
            },
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    env.close()


if __name__ == "__main__":
    main()
