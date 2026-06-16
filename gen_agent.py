import json
from pathlib import Path
from typing import Dict


def transcribe(file_path: str, api_key: str, model: str, prompt: str) -> Dict:
    try:
        import google.generativeai as genai
    except Exception as exc:
        return {"status": "error", "message": f"SDK import error: {exc}"}

    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    try:
        uploaded = genai.upload_file(path=file_path, mime_type="audio/wav", display_name=Path(file_path).name)
    except Exception as exc:
        return {"status": "error", "message": f"upload failed: {exc}"}

    try:
        model_obj = genai.GenerativeModel(model_name=model)
        response = model_obj.generate_content([prompt, uploaded], request_options={"timeout": 600})
    except Exception as exc:
        try:
            genai.delete_file(uploaded)
        except Exception:
            pass
        return {"status": "error", "message": f"generate failed: {exc}"}

    text = getattr(response, "text", "") or "\n".join(
        [getattr(c, "content", "") for c in (getattr(response, "candidates", []) or [])]
    )

    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            parsed["raw_text"] = text
            return {"status": "success", "result": parsed}
        except Exception:
            pass

    return {"status": "partial", "raw_text": text}


def transcribe_raw(file_path: str, api_key: str, model: str) -> Dict:
    """Return raw transcript text for the given audio file."""
    try:
        import google.generativeai as genai
    except Exception as exc:
        return {"status": "error", "message": f"SDK import error: {exc}"}

    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    try:
        uploaded = genai.upload_file(path=file_path, mime_type="audio/wav", display_name=Path(file_path).name)
    except Exception as exc:
        return {"status": "error", "message": f"upload failed: {exc}"}

    try:
        model_obj = genai.GenerativeModel(model_name=model)
        # Ask model to transcribe plainly
        response = model_obj.generate_content([f"Transcribe the audio file: {Path(file_path).name}", uploaded], request_options={"timeout": 600})
    except Exception as exc:
        try:
            genai.delete_file(uploaded)
        except Exception:
            pass
        return {"status": "error", "message": f"generate failed: {exc}"}

    text = getattr(response, "text", "") or "\n".join(
        [getattr(c, "content", "") for c in (getattr(response, "candidates", []) or [])]
    )

    try:
        genai.delete_file(uploaded)
    except Exception:
        pass

    return {"status": "success", "transcript": text}


def analyze_text(text: str, api_key: str, model: str, short_prompt: str) -> Dict:
    """Analyze a transcript text and return suggestions/business analysis JSON or partial raw text."""
    try:
        import google.generativeai as genai
    except Exception as exc:
        return {"status": "error", "message": f"SDK import error: {exc}"}

    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    try:
        model_obj = genai.GenerativeModel(model_name=model)
        response = model_obj.generate_content([short_prompt, text], request_options={"timeout": 600})
    except Exception as exc:
        return {"status": "error", "message": f"analysis failed: {exc}"}

    out = getattr(response, "text", "") or "\n".join(
        [getattr(c, "content", "") for c in (getattr(response, "candidates", []) or [])]
    )

    # try parse JSON
    cleaned = out.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            parsed["raw_text"] = out
            return {"status": "success", "analysis": parsed}
        except Exception:
            pass

    return {"status": "partial", "raw_text": out}


def analyze_meta(transcript: str, api_key: str, model: str) -> Dict:
    """Produce audio analysis and language identification based on the transcript."""
    try:
        import google.generativeai as genai
    except Exception as exc:
        return {"status": "error", "message": f"SDK import error: {exc}"}

    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    prompt = (
        "You are an audio analyst. Given the transcript below, provide:\n"
        "1) audio_analysis: describe waveform characteristics, background noise, speaker activity.\n"
        "2) language_identification: language name and confidence\n"
        "Return a JSON object with fields 'audio_analysis' and 'language_identification'.\n"
        "Transcript:\n" + transcript
    )

    try:
        model_obj = genai.GenerativeModel(model_name=model)
        response = model_obj.generate_content([prompt], request_options={"timeout": 300})
    except Exception as exc:
        return {"status": "error", "message": f"meta analysis failed: {exc}"}

    out = getattr(response, "text", "") or "\n".join(
        [getattr(c, "content", "") for c in (getattr(response, "candidates", []) or [])]
    )

    cleaned = out.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            parsed["raw_text"] = out
            return {"status": "success", "meta": parsed}
        except Exception:
            pass

    return {"status": "partial", "raw_text": out}
