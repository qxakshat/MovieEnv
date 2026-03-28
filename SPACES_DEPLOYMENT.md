# 🚀 Deploy to Hugging Face Spaces

This guide explains how to deploy the Movie Recommendation Agent to Hugging Face Spaces.

## Quick Start

1. **Create a new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `movie-recommendation-agent`
   - License: `openrail` (required for Llama)
   - SDK: **Gradio**
   - Space type: **Public** (or Private)

2. **Upload files:**
   - Clone or download this repo
   - Upload these files to your Space:
     - `app.py` (main Gradio app)
     - `requirements.txt` (dependencies)
     - `src/` folder (with `movie_recommendation_env.py`)

3. **Deploy:**
   - Hugging Face will automatically build and deploy
   - Wait for the build to complete
   - Your app will be live at: `https://huggingface.co/spaces/YOUR-USERNAME/movie-recommendation-agent`

## Files Required

```
movie-recommendation-agent/
├── app.py                          # Gradio interface
├── requirements.txt                # Python dependencies
└── src/
    ├── __init__.py
    ├── movie_recommendation_env.py # Environment code
    ├── data_loader.py
    └── system_prompt.txt
```

## Environment Variables (Optional)

No special environment variables are required. The app will:
- Auto-detect GPU if available
- Fall back to CPU for inference
- Cache the Llama model automatically

## Hardware Requirements

- **Recommended:** GPU with 4GB+ VRAM
- **Minimum:** CPU (slow, ~30-60sec per recommendation)
- Hugging Face Spaces provides:
  - **Free tier (CPU):** Limited resources
  - **Pro tier (GPU):** T4 GPU recommended

## Customization

Edit `app.py` to:
- Change the model: `MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"`
- Add more genres to the interface
- Modify the system prompt in `app.py`
- Add authentication/rate limiting

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Use CPU tier or reduce batch size |
| Model not loading | Ensure `meta-llama/Llama-3.2-1B-Instruct` is accessible |
| App crashes | Check logs in Space Settings → Logs |
| Recommendations are garbage | System prompt may be too long for inference-only mode |

## Local Testing

Before deploying to Spaces, test locally:

```bash
cd movieEnv
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860 in your browser.

## License

This Space uses Meta's Llama 3.2 model, which requires acceptance of the OpenRAIL license.

---

For more info on Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
