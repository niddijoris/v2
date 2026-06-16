import os
import json
import subprocess
import tempfile
from pathlib import Path

import streamlit as st


def load_dotenv(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def to_wav(input_path: str) -> str:
    out = str(Path(tempfile.mkdtemp()) / (Path(input_path).stem + ".wav"))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        out,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e}")
    return out


def call_gemini(wav_path: str, model: str, api_key: str, prompt: str) -> dict:
    try:
        import google.generativeai as genai
    except Exception as e:
        return {"status": "error", "message": f"Generative AI SDK not available: {e}"}

    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    try:
        uploaded = genai.upload_file(path=wav_path, mime_type="audio/wav", display_name=Path(wav_path).name)
    except Exception as e:
        return {"status": "error", "message": f"upload failed: {e}"}

    try:
        model_obj = genai.GenerativeModel(model_name=model)
        response = model_obj.generate_content([prompt, uploaded], request_options={"timeout": 600})
    except Exception as e:
        try:
            genai.delete_file(uploaded)
        except Exception:
            pass
        return {"status": "error", "message": f"generation failed: {e}"}

    text = ""
    text = getattr(response, "text", "") or "\n".join(
        [getattr(c, "content", "") for c in (getattr(response, "candidates", []) or [])]
    )

    # try extract JSON
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    parsed = None
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except Exception:
            parsed = None

    try:
        genai.delete_file(uploaded)
    except Exception:
        pass

    if parsed:
        parsed["raw_text"] = text
        return {"status": "success", "result": parsed}
    return {"status": "partial", "raw_text": text}


def build_prompt(file_name: str) -> str:
    return (
        "You are a Business Operations & Contract Analysis Agent. Analyze the provided audio and return a single JSON object matching the schema:"
        " meeting_summary, extracted_rates, operational_rules, risk_factors, action_items.\n"
        f"File: {file_name}\n"
        "If you cannot determine a field, set it to null or empty array. Provide concise business analysis."
    )


def main():
    st.set_page_config(page_title="Rebuilt Uzbek STT Agent", layout="wide")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL") or "gemini-1.5-pro"

    st.title("Rebuilt Uzbek STT — Single File")
    uploaded = st.file_uploader("Upload audio (.mp3, .m4a, .wav, .ogg)")
    if not uploaded:
        st.info("Drop an audio file to start")
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()

    try:
        wav = to_wav(tmp.name)
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return

    if not api_key:
        st.error("GEMINI_API_KEY not set in .env")
        return

    prompt = build_prompt(Path(uploaded.name).name)
    st.info("Uploading to Gemini and requesting analysis...")
    res = call_gemini(wav, model, api_key, prompt)

    if res.get("status") == "error":
        st.error(res.get("message"))
        return

    if res.get("status") == "partial":
        st.warning("Model returned non-JSON output. Showing raw text.")
        st.text_area("Raw output", res.get("raw_text", ""), height=400)
        return

    out = res.get("result", {})
    st.subheader("Meeting Summary")
    st.write(out.get("meeting_summary", ""))

    st.subheader("Extracted Rates")
    st.json(out.get("extracted_rates", {}))

    st.subheader("Operational Rules")
    st.json(out.get("operational_rules", []))

    st.subheader("Risk Factors")
    for r in out.get("risk_factors", []):
        st.write(f"- {r}")

    st.subheader("Action Items")
    st.json(out.get("action_items", []))


if __name__ == "__main__":
    main()
