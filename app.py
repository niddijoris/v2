"""Modular Streamlit entry — small and focused (<100 lines)."""
import tempfile
from pathlib import Path

import streamlit as st

from config import load_dotenv, get_api_key, get_model
from audio_simple import to_wav
from gen_agent import transcribe_raw, analyze_text


load_dotenv()
st.set_page_config(page_title="Modular Uzbek STT", layout="wide")


def main():
    st.title("Modular Uzbek STT")
    uploaded = st.file_uploader("Upload audio (.mp3,.m4a,.wav,.ogg)")
    if not uploaded:
        st.info("Upload a file to begin")
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()

    try:
        wav = to_wav(tmp.name)
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return

    api_key = get_api_key()
    model = get_model()
    if not api_key:
        st.error("GEMINI_API_KEY missing in .env")
        return

    # Step 1: Full transcript
    st.info("Requesting full transcript from Gemini...")
    t = transcribe_raw(wav, api_key, model)
    if t.get("status") == "error":
        st.error(t.get("message"))
        return
    transcript = t.get("transcript", "")
    st.subheader("Full Transcript")
    st.text_area("Transcript", transcript, height=360)

    # Step 2: Meta analysis (audio analysis + language identification)
    st.info("Requesting audio analysis and language identification...")
    from gen_agent import analyze_meta, analyze_text
    m = analyze_meta(transcript, api_key, model)
    if m.get("status") == "error":
        st.error(m.get("message"))
        return
    if m.get("status") == "partial":
        st.warning("Meta analysis returned non-JSON text; showing raw output.")
        st.text_area("Meta raw output", m.get("raw_text", ""), height=200)
    else:
        meta = m.get("meta", {})
        st.subheader("Audio Analysis")
        st.write(meta.get("audio_analysis", ""))
        st.subheader("Language Identification")
        st.write(meta.get("language_identification", {}))

    # Step 3: Business suggestions based on transcript
    analysis_prompt = (
        "You are a business analyst. Given the transcript below, return JSON with suggestions, action_items, and pain_points."
    )
    st.info("Requesting suggestions and action items...")
    a = analyze_text(transcript, api_key, model, analysis_prompt)
    if a.get("status") == "error":
        st.error(a.get("message"))
        return
    if a.get("status") == "partial":
        st.warning("Suggestions returned non-JSON text; showing raw output.")
        st.text_area("Suggestions raw output", a.get("raw_text", ""), height=300)
    else:
        analysis = a.get("analysis", {})
        st.subheader("Suggestions & Action Items")
        st.json(analysis)


if __name__ == "__main__":
    main()
