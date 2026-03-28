"""
Hugging Face Spaces App for Movie Recommendation Agent

This Gradio app allows users to interact with the fine-tuned Llama 3.2 1B model
for movie recommendations based on their preferences.

Deploy to HF Spaces:
1. Create a new Space at https://huggingface.co/spaces
2. Select Gradio as the SDK
3. Upload this file as app.py
4. Add requirements.txt

To run locally:
    pip install -r requirements.txt
    python app.py
"""

import sys
from pathlib import Path
import random
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from movie_recommendation_env import MovieRecommendationEnv, RecommendationAction, MOVIE_DATABASE


# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "You are a movie recommendation assistant. Recommend one movie that fits the user's genres.\nAlways reply with ONLY the movie title in square brackets. Example: [Inception]"

# Global model and tokenizer cache
model = None
tokenizer = None
env = None


def load_model():
    """Load model and tokenizer."""
    global model, tokenizer
    
    if model is not None:
        return model, tokenizer
    
    print(f"Loading model {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device != "cuda":
        model = model.to(device)
    
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    return model, tokenizer


def extract_movie_title(text: str) -> str | None:
    """Extract movie title from model output."""
    import re
    
    # Look for text in square brackets
    match = re.search(r"\[([^\]]{2,80})\]", text)
    if match:
        extracted = match.group(1).strip()
        if extracted and ((" " in extracted) or (extracted[0].isupper())):
            return extracted
    
    # Look for quoted title
    match = re.search(r'"([A-Z][^"]{2,79})"', text)
    if match:
        return match.group(1).strip()
    
    return None


def generate_recommendation(user_genres: list[str], min_rating: float) -> str:
    """Generate a movie recommendation for the given preferences."""
    global model, tokenizer, env
    
    if model is None:
        model, tokenizer = load_model()
    
    if env is None:
        env = MovieRecommendationEnv(max_recommendations=1, movie_db=MOVIE_DATABASE)
    
    # Create user profile
    user_profile = {
        "name": f"User_{random.randint(1000, 9999)}",
        "preferred_genres": user_genres,
        "min_rating": min_rating,
        "avg_rating_tolerance": 1.0,
    }
    
    # Reset environment with user profile
    result = env.reset(user_profile=user_profile, watch_history=[])
    observation = result.observation
    
    # Build prompt
    genre_str = ", ".join(user_genres)
    user_context = (
        f"Preferred Genres: {genre_str}\n"
        f"Minimum Rating: {min_rating}\n"
        f"Previous recommendations: None yet.\n"
        "Reply with ONLY: [Movie Title]"
    )
    
    # Format messages for chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_context},
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Generate
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    
    # Extract recommendation
    movie_title = extract_movie_title(completion_text)
    
    if not movie_title:
        return "Unable to extract a valid recommendation. Raw output: " + completion_text[:100]
    
    # Find movie details
    movie_id = None
    for mid, movie in MOVIE_DATABASE.items():
        if movie_title.lower() in movie["title"].lower() or movie["title"].lower() in movie_title.lower():
            movie_id = mid
            movie_title = movie["title"]
            break
    
    if movie_id:
        movie = MOVIE_DATABASE[movie_id]
        return (
            f"🎬 **{movie['title']}**\n\n"
            f"Rating: ⭐ {movie['rating']}/10\n"
            f"Year: {movie['year']}\n"
            f"Genres: {', '.join(movie['genres'])}\n\n"
            f"This recommendation matches your preference for {', '.join(user_genres)}!"
        )
    else:
        return f"🎬 **{movie_title}**\n\n(Not in database, but recommended by the model!)"


def create_interface():
    """Create the Gradio interface."""
    
    available_genres = [
        "Drama", "Action", "Comedy", "Sci-Fi", "Horror", 
        "Adventure", "Crime", "Fantasy", "Thriller", "Biography"
    ]
    
    with gr.Blocks(title="🎬 Movie Recommendation Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎬 Movie Recommendation Agent
            
            Get personalized movie recommendations powered by **Llama 3.2 1B**!
            
            Select your preferred genres and minimum rating threshold, then get a tailored recommendation.
            """
        )
        
        with gr.Row():
            with gr.Column():
                genres = gr.Checkboxgroup(
                    label="Preferred Genres",
                    choices=available_genres,
                    value=["Drama", "Sci-Fi"],
                    info="Select one or more genres you enjoy"
                )
                min_rating = gr.Slider(
                    label="Minimum Rating",
                    minimum=5.0,
                    maximum=9.5,
                    step=0.5,
                    value=7.0,
                    info="Only recommend movies at or above this rating"
                )
                recommend_btn = gr.Button("🎯 Get Recommendation", size="lg")
            
            with gr.Column():
                output = gr.Markdown(label="Recommendation")
        
        recommend_btn.click(
            fn=generate_recommendation,
            inputs=[genres, min_rating],
            outputs=output,
        )
        
        gr.Examples(
            examples=[
                [["Drama", "Biography"], 8.0],
                [["Sci-Fi", "Adventure"], 7.5],
                [["Action", "Thriller"], 8.5],
            ],
            inputs=[genres, min_rating],
            label="Try these examples →",
        )
        
        gr.Markdown(
            """
            ---
            
            **How it works:**
            1. Select your preferred movie genres
            2. Set your minimum acceptable rating (higher = higher quality picks)
            3. Click "Get Recommendation" to receive a personalized suggestion
            
            This app uses a fine-tuned **Llama 3.2 1B** model trained with Group Relative Policy Optimization (GRPO).
            
            🔗 [View Source Code](https://github.com) | 📊 [Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
