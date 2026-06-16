# Uzbek STT Pseudo-Transcription (Streamlit)

Quick scaffold for a Streamlit UI that preprocesses audio and calls a multimodal LLM for "pseudo-transcription" in Uzbek.

Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system (Homebrew: `brew install ffmpeg`)

Install dependencies
```bash
python -m pip install -r requirements.txt
```

Run the app
```bash
streamlit run app.py
```

Notes
- The `gemini_agent` contains a placeholder integration. Set `GEMINI_API_KEY` in `st.secrets` or environment to enable real API calls, and replace the pseudocode with your project's GenAI SDK usage.
- After processing, uploaded temporary files are left for inspection; you may add cleanup as desired.
