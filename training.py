"""
GRPO Training Script for Movie Recommendation Agent

This script trains a language model to generate better movie recommendations
using Group Relative Policy Optimization (GRPO) via TRL.

Based on Meta's OpenEnv and Hugging Face's TRL framework.

Usage:
    python training.py --model-id meta-llama/Llama-3.2-1B-Instruct --num-epochs 3
    
    # With device selection
    python training.py --device cuda    # NVIDIA GPU (fastest)
    python training.py --device mps     # Apple Silicon Mac
    python training.py --device cpu     # CPU only
    
    # Auto-detects best available device if --device not specified

Requirements:
    - transformers
    - peft
    - torch >= 2.0.0
    - datasets

Device Support:
    - CUDA: NVIDIA GPUs (recommended for training)
    - MPS: Apple Silicon Macs (use --per-device-batch-size 1)
    - CPU: Fallback (slow, for testing only)
"""

from __future__ import annotations

import argparse
import sys
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recommendation_env import (
    MovieRecommendationEnv,
    RecommendationAction,
    Message,
    MOVIE_DATABASE,
)
from data_loader import MovieDatabase


# ============================================================================
# Configuration
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a movie recommendation agent using GRPO and OpenEnv"
    )

    # Model configuration
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model identifier for fine-tuning (Hugging Face model hub)",
    )
    parser.add_argument(
        "--tokenizer-id",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Tokenizer identifier (usually same as model)",
    )

    # Environment configuration
    parser.add_argument(
        "--max-recommendations",
        type=int,
        default=6,
        help="Maximum recommendations per episode",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=1000,
        help="Number of users in synthetic training dataset",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for GRPO training",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=64,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="Number of rollout generations per dataset prompt",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )

    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum new tokens for generation",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Device to use for training (auto/cuda/cpu/mps)",
    )
    parser.add_argument(
        "--movie-database",
        default="movielens_100k",
        help="Movie database to use (default: movielens_100k). "
             "Can be 'movielens_100k' (1,682 movies), 'movielens_1m' (3,883 movies), "
             "or path to custom JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Checkpoint save interval",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name for tracking",
    )

    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================


def resolve_system_prompt(path: str) -> str:
    """Load system prompt from file."""
    prompt_path = Path(path)
    if not prompt_path.is_file():
        prompt_path = Path(__file__).parent / "src" / path
    if prompt_path.is_file():
        content = prompt_path.read_text()
        print(f"✓ System prompt loaded from {prompt_path} ({len(content)} chars)", flush=True)
        return content
    
    print(f"⚠️  System prompt file not found at {path}, using default", flush=True)
    return "You are a helpful movie recommendation assistant. Recommend movies based on user preferences."


def sanitize_name(name: str) -> str:
    """Sanitize names for file paths."""
    return name.replace("/", "-").replace(" ", "_")


def format_history(messages: Iterable[Message]) -> str:
    """Format message history for model input — only recommendations and feedback."""
    lines = []
    for message in messages:
        # Skip PROMPT and USER_PROFILE to avoid duplicating context already in system prompt
        if message.category not in ("RECOMMENDATION", "FEEDBACK"):
            continue
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"{message.category}: {content[:80]}")
    return "\n".join(lines[-6:])  # Keep last 6 turns at most


def make_user_prompt(prompt_text: str, messages: Iterable[Message]) -> str:
    """Build a compact user prompt - extract key user info and recent history only."""
    # Extract genre and rating from prompt_text (avoid passing whole profile)
    genre_line = ""
    rating_line = ""
    for line in prompt_text.split("\n"):
        if "Preferred Genres" in line and not genre_line:
            genre_line = line.strip()
        elif "Minimum Rating" in line and not rating_line:
            rating_line = line.strip()
    user_info = ", ".join(filter(None, [genre_line, rating_line])) or "Various genres"

    history = format_history(messages)
    history_section = history if history else "None yet."

    return (
        f"{user_info}\n"
        f"Previous recommendations:\n{history_section}\n"
        "Reply with ONLY: [Movie Title]"
    )


def extract_movie_recommendation(text: str) -> str | None:
    """Extract movie recommendation from model output."""
    import re

    # Try 1: Look for text in square brackets
    match = re.search(r"\[([^\]]{2,80})\]", text)
    if match:
        extracted = match.group(1).strip()
        # Reject if no spaces and all lowercase (likely a garbage concatenation from prompt echo)
        if extracted and (" " in extracted or extracted[0].isupper()):
            return extracted

    # Try 2: Quoted title
    match = re.search(r'"([A-Z][^"]{2,79})"', text)
    if match:
        return match.group(1).strip()

    # No fallback to raw text — too risky for a small model
    return None


def scale_quality_score(rating: float, max_rating: float = 10.0) -> float:
    """Scale movie rating to 0-1 score."""
    return min(1.0, rating / max_rating)


# ============================================================================
# Reward Functions
# ============================================================================


def reward_genre_match(completions: list[str], **kwargs) -> list[float]:
    """Reward for genre matching."""
    rewards = kwargs.get("genre_match_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_quality(completions: list[str], **kwargs) -> list[float]:
    """Reward for movie quality (rating)."""
    rewards = kwargs.get("quality_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_no_repeat(completions: list[str], **kwargs) -> list[float]:
    """Reward for not recommending duplicates."""
    rewards = kwargs.get("no_repeat_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_overall_success(completions: list[str], **kwargs) -> list[float]:
    """Overall success reward."""
    rewards = kwargs.get("overall_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ============================================================================
# Generation (Direct inference without vLLM)
# ============================================================================


def generate_completions_fallback(
    trainer: GRPOTrainer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 10,
    top_p: float = 0.9,
) -> dict:
    """Direct completion generation using model inference."""
    # Ensure model is in eval mode
    trainer.model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    
    # Generate with the model - use more stable parameters
    try:
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 16),  # Reduce max tokens to avoid runaway
                temperature=0.7,  # Slightly lower for stability
                top_k=5,  # More conservative
                top_p=0.9,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Penalize repetition
            )
    except Exception as e:
        print(f"  [ERROR] Generation failed: {e}", flush=True)
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": [],
            "logprobs": [],
            "text": "",
        }
    
    completion_ids = outputs.sequences[0][len(prompt_ids):].tolist()
    
    # Default log probabilities (since we disabled output_scores)
    logprobs = [0.0] * len(completion_ids)
    
    # Decode completion text with error handling
    try:
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  [ERROR] Decoding failed: {e}", flush=True)
        text = ""
    
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": text,
    }


# ============================================================================
# Environment Interaction
# ============================================================================


def rollout_once(
    trainer: GRPOTrainer,
    env: MovieRecommendationEnv,
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
) -> dict[str, list]:
    """
    Execute one full recommendation episode.
    
    Returns dict with prompt_ids, completion_ids, logprobs, and reward signals.
    """
    # Validate inputs
    if not system_prompt or len(system_prompt) < 10:
        print(f"  [WARNING] System prompt seems too short: {len(system_prompt)} chars", flush=True)
    
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    genre_match_rewards: list[float] = []
    quality_rewards: list[float] = []
    no_repeat_rewards: list[float] = []
    overall_rewards: list[float] = []
    recommendations_made: set[str] = set()

    for turn in range(max_turns):
        # Check if episode is done (only after env.step())
        if turn > 0 and hasattr(result, 'done') and result.done:
            break

        # Prepare prompt
        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Debug: Print prompts on first turn only
        if turn == 0:
            print(f"  [DEBUG PROMPT] System prompt (first 150 chars): {system_prompt[:150]!r}", flush=True)
            print(f"  [DEBUG PROMPT] User prompt (first 200 chars): {user_prompt[:200]!r}", flush=True)
            print(f"  [DEBUG PROMPT] Full prompt_text length: {len(prompt_text)} chars", flush=True)
            print(f"  [DEBUG PROMPT] First 300 chars of formatted prompt: {prompt_text[:300]!r}", flush=True)

        # Generate recommendation using direct inference
        if turn == 0:
            print(f"  [DEBUG] Using direct model generation", flush=True)
        
        rollout_outputs = generate_completions_fallback(
            trainer=trainer,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=32,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )
        
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )
        
        # Debug: Show completion quality on first turn
        if turn == 0:
            print(f"  [DEBUG] Completion IDs count: {len(rollout_outputs['completion_ids'])}", flush=True)
            print(f"  [DEBUG] Raw completion IDs (first 10): {rollout_outputs['completion_ids'][:10]}", flush=True)
            print(f"  [DEBUG] Completion text length: {len(completion_text)}", flush=True)
            print(f"  [DEBUG] Full completion text: {completion_text!r}", flush=True)
            if not completion_text.strip():
                print(f"  [WARNING] Empty completion text! This suggests model generation is failing.", flush=True)

        # Extract and validate recommendation
        movie_rec = extract_movie_recommendation(completion_text)
        
        # Debug: print raw output and extraction attempt
        if turn == 0:  # Print details only on first turn to avoid spam
            print(f"  [DEBUG EXTRACTION] Raw completion_text (first 200 chars): {completion_text[:200]!r}", flush=True)
            print(f"  [DEBUG EXTRACTION] Extracted movie: {movie_rec}", flush=True)
        
        if not movie_rec:
            print(f"  [DEBUG] turn {turn+1}: Extraction FAILED, using fallback. Raw completion: {completion_text[:100]!r}", flush=True)
            movie_rec = "The Shawshank Redemption"  # fallback

        # Step environment
        action = RecommendationAction(movie_title=movie_rec)
        result = env.step(action)
        observation = result.observation

        print(f"  [DEBUG] turn {turn+1}: movie='{movie_rec}' reward={result.reward:.3f}", flush=True)

        # Process rewards
        if movie_rec in recommendations_made:
            no_repeat_rewards.append(-0.5)
        else:
            no_repeat_rewards.append(0.1)
        recommendations_made.add(movie_rec)

        # Extract reward components
        reward_breakdown = result.info.get("reward_breakdown", {})
        genre_match_rewards.append(reward_breakdown.get("genre_match", 0.0))
        quality_rewards.append(reward_breakdown.get("rating_quality", 0.0))
        overall_rewards.append(result.reward)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "genre_match_reward": genre_match_rewards,
        "quality_reward": quality_rewards,
        "no_repeat_reward": no_repeat_rewards,
        "overall_reward": overall_rewards,
    }


# ============================================================================
# Dataset Loading
# ============================================================================


def load_movie_database(db_arg):
    """Load movie database from various sources.
    
    Args:
        db_arg: None (built-in), file path, or 'movielens_100k'/'movielens_1m'
        
    Returns:
        Movie database dictionary
    """
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


# ============================================================================
# Main Training
# ============================================================================


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


def main() -> None:
    """Main training loop."""
    args = parse_args()
    
    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}\n")

    # Initialize model and tokenizer
    print(f"Loading tokenizer from {args.tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load movie database
    movie_db = load_movie_database(args.movie_database)
    print(f"Loaded {len(movie_db)} movies for training\n")

    # Initialize environment
    print("Initializing Movie Recommendation environment...")
    env = MovieRecommendationEnv(
        max_recommendations=args.max_recommendations,
        movie_db=movie_db
    )

    # Load system prompt
    system_prompt = resolve_system_prompt("system_prompt.txt")

    # Create training dataset
    print(f"Creating synthetic dataset with {args.max_users} users...")
    dataset_prompt = "Recommend movies the user will enjoy."
    dataset = Dataset.from_dict({"prompt": [dataset_prompt] * args.max_users})

    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"movie-rec-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GRPO Configuration
    # Note: Device selection (CUDA/MPS/CPU) is handled automatically
    # For MPS, disable pin_memory as it's not supported
    
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=20,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        report_to="none",
        save_strategy="steps",
        save_steps=args.save_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        dataloader_pin_memory=False if device == "mps" else True,
    )

    grpo_config.run_name = args.run_name or f"movie-rec-{timestamp}"

    # Define rollout function
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        """Rollout function for GRPO training."""
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        episode_genre_rewards: list[list[float]] = []
        episode_quality_rewards: list[list[float]] = []
        episode_repeat_rewards: list[list[float]] = []
        episode_overall_rewards: list[list[float]] = []

        print(f"[DEBUG] rollout_func called with {len(prompts)} prompts", flush=True)
        for i, prompt_text in enumerate(prompts):
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                dataset_prompt=prompt_text,
                system_prompt=system_prompt,
                max_turns=args.max_recommendations,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            episode_genre_rewards.append(episode["genre_match_reward"])
            episode_quality_rewards.append(episode["quality_reward"])
            episode_repeat_rewards.append(episode["no_repeat_reward"])
            episode_overall_rewards.append(episode["overall_reward"])
            print(f"[DEBUG] episode {i+1}/{len(prompts)} done — turns={len(episode['overall_reward'])}, avg_reward={sum(episode['overall_reward'])/max(1,len(episode['overall_reward'])):.3f}", flush=True)

        # Flatten nested reward lists for reward functions
        flat_genre_rewards = [r for rewards_list in episode_genre_rewards for r in rewards_list]
        flat_quality_rewards = [r for rewards_list in episode_quality_rewards for r in rewards_list]
        flat_repeat_rewards = [r for rewards_list in episode_repeat_rewards for r in rewards_list]
        flat_overall_rewards = [r for rewards_list in episode_overall_rewards for r in rewards_list]
        
        # Debug: Show shapes
        print(f"[DEBUG ROLLOUT] Flattened reward shapes - genre: {len(flat_genre_rewards)}, quality: {len(flat_quality_rewards)}, repeat: {len(flat_repeat_rewards)}, overall: {len(flat_overall_rewards)}", flush=True)

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "genre_match_reward": flat_genre_rewards,
            "quality_reward": flat_quality_rewards,
            "no_repeat_reward": flat_repeat_rewards,
            "overall_reward": flat_overall_rewards,
        }

    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_genre_match,
            reward_quality,
            reward_no_repeat,
            reward_overall_success,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )
    
    # Validate model setup
    print(f"✓ Model device: {trainer.model.device}", flush=True)
    print(f"✓ Model dtype: {trainer.model.dtype}", flush=True)
    print(f"✓ Tokenizer vocab size: {tokenizer.vocab_size}", flush=True)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print(f"⚠️  WARNING: Tokenizer has no chat_template, may affect formatting", flush=True)

    # Train
    print("\n" + "=" * 70)
    print("STARTING GRPO TRAINING FOR MOVIE RECOMMENDATION AGENT")
    print("=" * 70)
    print(f"Model: {args.model_id}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 70 + "\n")

    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"Checkpoints saved to: {output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
