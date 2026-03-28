# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Run the Simple Example** (no GPU needed)

```bash
python examples_simple.py
```

This will test the environment without any ML:

```
🎬 SIMPLE MOVIE RECOMMENDATION ENVIRONMENT DEMO
==================================================================================
User: Alice
Preferred Genres: Drama, Crime
Minimum Rating: 8.0
Watch History: 2 movies

Step 1/6: Recommending The Shawshank Redemption
✅ Success!
   Title: The Shawshank Redemption
   Reward: 0.850
   Genre Match: 0.500
   Quality Score: 0.930
```

### 3. **Explore with Jupyter** (interactive)

```bash
jupyter notebook notebooks/MovieRecommendationEnv_Tutorial.ipynb
```

This notebook shows:
- Environment exploration
- Dataset visualization
- Manual recommendation testing
- Batch user simulation
- Reward analysis with plots

### 4. **Train a Model** (auto-detects GPU: CUDA > MPS > CPU)

```bash
# Basic training (auto-detect device, ~10 min on GPU)
python training.py \
  --model-id Qwen/Qwen3-1.7B \
  --num-epochs 3 \
  --max-users 100

# Specify device explicitly
python training.py --device cuda ...    # NVIDIA GPU (fastest)
python training.py --device mps ...     # Apple Silicon Mac
python training.py --device cpu ...     # CPU only (slow)

# Advanced training with custom parameters
python training.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --learning-rate 5e-6 \
  --num-epochs 5 \
  --max-users 1000 \
  --temperature 0.8 \
  --device mps
```

**Key Parameters:**
- `--model-id`: HuggingFace model ID for fine-tuning
- `--device`: cuda/mps/cpu (default: auto-detect)
- `--learning-rate`: 5e-6 to 1e-5 (smaller = more stable)
- `--num-epochs`: 1-5 (more = better but slower)
- `--max-users`: 100-5000 (more users = better generalization)
- `--temperature`: 0.7-0.9 (lower = more deterministic)

### 5. **Run Inference**

```bash
# Generate recommendations (auto-detect device)
python inference.py \
  --model-id Qwen/Qwen3-1.7B \
  --user-genres "Drama,Crime" \
  --min-rating 8.0 \
  --num-recommendations 6

# With device specification
python inference.py \
  --model-id path/to/trained/model \
  --user-genres "Sci-Fi,Action" \
  --device cuda         # NVIDIA GPU
  # or --device mps     # Apple Silicon
  # or --device cpu     # CPU

# With custom parameters
python inference.py \
  --model-id path/to/trained/model \
  --user-genres "Drama,Crime" \
  --min-rating 8.0 \
  --num-recommendations 6 \
  --temperature 0.9 \
  --device mps
```

---

## 📊 Project Structure

```
MovieRecommendationEnv/
├── src/
│   ├── movie_recommendation_env.py   # Core environment (read this!)
│   └── system_prompt.txt              # Agent instructions
├── notebooks/
│   └── MovieRecommendationEnv_Tutorial.ipynb  # Interactive examples
├── training.py                        # GRPO training
├── inference.py                       # Recommendation generation
├── examples_simple.py                 # Simple test
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker config
└── README.md                          # Full documentation
```

---

## 🖥️ GPU Device Support

Both `training.py` and `inference.py` support multiple compute devices:

### Device Options

| Device | Command | Best For | Speed |
|--------|---------|----------|-------|
| Auto-detect | (default) | Smart selection | ⚡⚡⚡ |
| NVIDIA CUDA | `--device cuda` | NVIDIA GPUs | ⚡⚡⚡ |
| Apple Metal (MPS) | `--device mps` | Apple Silicon Mac | ⚡⚡ |
| CPU | `--device cpu` | No GPU available | ⚡ |

### Check Available Devices

```python
import torch

print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS:  {torch.backends.mps.is_available()}")
print(f"CPU:  Available")

# Auto-detection
if torch.cuda.is_available():
    print("→ Use: CUDA (fastest)")
elif torch.backends.mps.is_available():
    print("→ Use: MPS (Apple Silicon)")
else:
    print("→ Use: CPU")
```

### MPS (Apple Silicon) Tips

If using a Mac with Apple Silicon:

```bash
# Use smaller batch size for MPS
python training.py --device mps --per-device-batch-size 1

# Inference with MPS
python inference.py --device mps --user-genres "Drama,Crime"

# Check if MPS is working
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

---

## 🎯 Common Tasks

### View MovieDatabase

```python
from src.movie_recommendation_env import MOVIE_DATABASE, GENRE_MAPPING

print(f"Total movies: {len(MOVIE_DATABASE)}")
print(f"Available genres: {list(GENRE_MAPPING.keys())}")

for movie_id, movie in list(MOVIE_DATABASE.items())[:3]:
    print(f"{movie['title']} ({movie['rating']}) - {movie['genres']}")
```

### Create Custom User Profile

```python
from src.movie_recommendation_env import MovieRecommendationEnv, RecommendationAction

env = MovieRecommendationEnv(max_recommendations=6)

user_profile = {
    "name": "CustomUser",
    "preferred_genres": ["Sci-Fi", "Action"],
    "min_rating": 7.5,
    "avg_rating_tolerance": 1.0,
}

result = env.reset(user_profile=user_profile)
```

### Test a Recommendation

```python
action = RecommendationAction(movie_id="tt0111161")
result = env.step(action)

print(f"Reward: {result.reward}")
print(f"Info: {result.info}")
```

---

## 🐳 Docker Usage

### Build Docker Image

```bash
docker build -t movie-recommendation-env:latest .
```

### Run Training in Docker

```bash
docker run --gpus all -it \
  -v $(pwd)/outputs:/app/outputs \
  movie-recommendation-env:latest \
  python training.py --num-epochs 3
```

### Docker Compose (Full Stack)

```bash
# Start training service + Jupyter
docker-compose up -d

# View logs
docker-compose logs -f movie-rec-env

# Access Jupyter at http://localhost:8888
```

---

## 💡 Tips & Tricks

### Speed Up Training

1. Reduce `--max-users` for faster iteration
2. Use smaller model (`Qwen/Qwen3-1.7B` instead of larger)
3. Run on GPU: ensure `CUDA_VISIBLE_DEVICES` is set
4. Use mixed precision: `--use-cache` flag in TRL

### Better Recommendations

1. Train with more data (`--max-users 5000+`)
2. Increase epochs (`--num-epochs 5`)
3. Tune temperature (0.6-0.8 for more deterministic)
4. Add more movies to database (edit `MOVIE_DATABASE`)

### Debug Issues

- Check logs: `tensorboard --logdir outputs/`
- Print verbose: Add `--debug` flag
- Test environment: Run `examples_simple.py` first
- Check memory: `nvidia-smi` (for GPU)

---

## 📚 Learning Resources

- **OpenEnv Tutorial**: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial
- **TRL Documentation**: https://huggingface.co/docs/trl/index
- **Gymnasium API**: https://gymnasium.farama.org/
- **GRPO Paper**: Group Relative Policy Optimization

---

## ❓ FAQ

**Q: Do I need a GPU?**
A: No for examples and simple tests. Yes for training (training without GPU will be very slow).

**Q: Can I use my Mac with Apple Silicon?**
A: Yes! Use `--device mps` for GPU acceleration on Apple Silicon Macs. The script auto-detects MPS availability.

```bash
python training.py --device mps --model-id Qwen/Qwen3-1.7B --per-device-batch-size 1
python inference.py --device mps --user-genres Drama,Crime
```

**Q: Which model should I use?**
A: Start with `Qwen/Qwen3-1.7B` (fast), then try larger models like `Llama-2-7b`.

**Q: How long does training take?**
A: ~10-30 minutes per epoch on GPU, ~2-3 minutes on MPS, much longer on CPU.

**Q: Can I use my own movies?**
A: Yes! Edit `MOVIE_DATABASE` in `src/movie_recommendation_env.py`.

**Q: How do I evaluate the model?**
A: Use `inference.py` and check the `recommendation_results.json` output.

---

## 🎬 Next Steps

1. ✅ Run `python examples_simple.py` to verify setup
2. ✅ Explore `notebooks/MovieRecommendationEnv_Tutorial.ipynb`
3. ✅ Train a model: `python training.py`
4. ✅ Generate recommendations: `python inference.py`
5. ✅ Submit to Meta PyTorch Hackathon! 🚀

---

**Happy coding! 🎬✨**
