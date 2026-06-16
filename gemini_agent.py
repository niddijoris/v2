import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


UZBEK_SYSTEM_PROMPT = (
    "Siz audio tahlil qilish bo'yicha yuqori malakali, o'zbek tilida so'zlashuvchi Multimodal Ekspertsiz.\n"
    "Sizga yuklangan audio fayl ichida fon shovqinlari, ovoz aks-sadolari (echo) va bir vaqtning o'zida gapiradigan 2-3 kishi bo'lishi mumkin.\n"
    "Vazifa: An'anaviy transkripsiya qilmang (so'zma-so'z xatolarga chalinmang). Ma'noni va dialoq zanjirini saqlagan holda \"Pseudo-transcription\" yarating. Agar yangi ovoz paydo bo'lsa, uni \"Speaker N\" deb belgilang. Fon shovqinlarini matnga qo'shmang. Natijani faqat JSON sxemasiga mos holda qaytaring.\n"
    "Agar audio biror qismida noaniqlik bo'lsa, pain_points maydonida qisqa va aniq sabablarni yozing.\n"
    "business_analysis maydonida qisqa xulosa, biznes qiymati, xavflar, imkoniyatlar va tavsiyalarni bering.\n"
    "completeness maydonida percent, is_complete, missing_sections ni qaytaring."
)


def _dedupe(items: List[Any]) -> List[Any]:
    seen = set()
    result: List[Any] = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else item
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _strip_json_wrappers(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _mime_type_for_path(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "audio/wav"


class GeminiAgent:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.2):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Allow overriding the model via argument or environment variable GEMINI_MODEL
        self.model_name = model or os.getenv("GEMINI_MODEL") or "gemini-1.5-pro"
        self.temperature = temperature
        self.genai = None
        self.model = None

        if not self.api_key:
            return

        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai SDK is not installed or could not be imported") from exc

        genai.configure(api_key=self.api_key)
        self.genai = genai
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "response_mime_type": "application/json",
            },
            system_instruction=UZBEK_SYSTEM_PROMPT,
        )

    def _empty_result(self, file_path: str, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "source_file": Path(file_path).name,
            "meta": {
                "detected_speakers_count": 0,
                "audio_quality_assessment": "unknown",
                "completeness": {
                    "percent": 0,
                    "is_complete": False,
                    "missing_sections": ["transcript", "summary", "action_items"],
                },
            },
            "transcript": [],
            "summary": "",
            "action_items": [],
            "pain_points": [message],
            "business_analysis": {
                "overview": "Audio could not be analyzed because the Gemini request failed.",
                "pain_points": [message],
                "risks": ["No reliable transcription was produced."],
                "opportunities": ["Retry after fixing the SDK or API key configuration."],
                "recommendations": ["Validate the Gemini API key, network access, and model quota."],
                "next_steps": ["Re-run the upload after the API issue is resolved."],
            },
        }

    def _offline_result(self, file_path: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "source_file": Path(file_path).name,
            "meta": {
                "detected_speakers_count": 2,
                "audio_quality_assessment": "Offline mock response; API key not set.",
                "completeness": {
                    "percent": 35,
                    "is_complete": False,
                    "missing_sections": ["verified_transcript", "business_context"],
                },
            },
            "transcript": [
                {"speaker": "Speaker 1", "text": "Salom, bu sinov transkripti.", "timestamp": "00:00"},
            ],
            "summary": "Offline mock summary. Add a Gemini API key to enable real transcription.",
            "action_items": ["Set GEMINI_API_KEY and retry with a real audio file."],
            "pain_points": ["API key is missing, so the real model cannot be called."],
            "business_analysis": {
                "overview": "This is a local mock response only.",
                "pain_points": ["No real speech analysis was performed."],
                "risks": ["Users may mistake the mock output for a production transcript."],
                "opportunities": ["Enable the Gemini API to produce a real business-ready transcript."],
                "recommendations": ["Configure the API key in .env before using production data."],
                "next_steps": ["Run the Streamlit app again after setting the key."],
            },
        }

    def _prompt(self, file_path: str) -> str:
        return (
            "Quyidagi audio faylni tahlil qiling va faqat JSON qaytaring. "
            "Pseudo-transcription qiling, speakerlarni ajrating, vaqt belgilari qo'ying, va so'zma-so'z xatolardan saqlaning. "
            "Agar matn qismi noaniq bo'lsa, pain_points va completeness.missing_sections ni to'ldiring. "
            "business_analysis ichida qisqa xulosa, biznes qiymati, asosiy xavflar, imkoniyatlar va tavsiyalarni bering. "
            f"Fayl nomi: {Path(file_path).name}."
        )

    def _response_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text)
        candidates = getattr(response, "candidates", None) or []
        parts: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                piece = getattr(part, "text", None)
                if piece:
                    parts.append(str(piece))
        return "\n".join(parts)

    def _normalize_result(self, data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        meta = data.get("meta") or {}
        completeness = meta.get("completeness") or {}
        transcript = data.get("transcript") or []
        action_items = data.get("action_items") or []
        pain_points = data.get("pain_points") or []
        business_analysis = data.get("business_analysis") or {}

        if not isinstance(transcript, list):
            transcript = []
        if not isinstance(action_items, list):
            action_items = [str(action_items)] if action_items else []
        if not isinstance(pain_points, list):
            pain_points = [str(pain_points)] if pain_points else []
        if not isinstance(business_analysis, dict):
            business_analysis = {"overview": str(business_analysis)}

        normalized = {
            "status": data.get("status") or "success",
            "source_file": Path(file_path).name,
            "meta": {
                "detected_speakers_count": int(meta.get("detected_speakers_count") or 0),
                "audio_quality_assessment": meta.get("audio_quality_assessment") or "unknown",
                "completeness": {
                    "percent": int(completeness.get("percent") or (100 if transcript else 0)),
                    "is_complete": bool(completeness.get("is_complete") if completeness else bool(transcript)),
                    "missing_sections": completeness.get("missing_sections") or [],
                },
            },
            "transcript": transcript,
            "summary": data.get("summary") or "",
            "action_items": _dedupe(action_items),
            "pain_points": _dedupe(pain_points),
            "business_analysis": {
                "overview": business_analysis.get("overview") or data.get("summary") or "",
                "pain_points": _dedupe(business_analysis.get("pain_points") or pain_points),
                "risks": _dedupe(business_analysis.get("risks") or []),
                "opportunities": _dedupe(business_analysis.get("opportunities") or []),
                "recommendations": _dedupe(business_analysis.get("recommendations") or []),
                "next_steps": _dedupe(business_analysis.get("next_steps") or []),
            },
        }

        if not normalized["meta"]["completeness"]["missing_sections"] and normalized["meta"]["completeness"]["percent"] < 100:
            normalized["meta"]["completeness"]["missing_sections"] = ["some_transcript_segments"]

        if normalized["meta"]["completeness"]["percent"] < 100 and normalized["status"] == "success":
            normalized["status"] = "partial_success"

        if not normalized["pain_points"] and normalized["meta"]["completeness"]["percent"] < 100:
            normalized["pain_points"] = ["Audio contains ambiguous or incomplete segments that need manual review."]

        if not normalized["business_analysis"]["pain_points"]:
            normalized["business_analysis"]["pain_points"] = normalized["pain_points"]

        if not normalized["business_analysis"]["overview"]:
            normalized["business_analysis"]["overview"] = normalized["summary"]

        return normalized

    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """Transcribe the audio file using Gemini File API."""
        if not self.api_key:
            return self._offline_result(file_path)

        if self.genai is None or self.model is None:
            return self._empty_result(file_path, "GenAI SDK is not initialized.")

        uploaded_file = None
        try:
            uploaded_file = self.genai.upload_file(
                path=file_path,
                mime_type=_mime_type_for_path(file_path),
                display_name=Path(file_path).stem,
            )
            response = self.model.generate_content(
                [self._prompt(file_path), uploaded_file],
                request_options={"timeout": 600},
            )
            raw_text = self._response_text(response)
            if not raw_text.strip():
                return self._empty_result(file_path, "Gemini returned an empty response.")

            parsed_text = _strip_json_wrappers(raw_text)
            parsed = json.loads(parsed_text)
            if not isinstance(parsed, dict):
                return self._empty_result(file_path, "Gemini returned JSON, but it was not an object.")

            return self._normalize_result(parsed, file_path)
        except json.JSONDecodeError:
            return self._empty_result(file_path, "Gemini output was not valid JSON.")
        except Exception as exc:
            return self._empty_result(file_path, str(exc))
        finally:
            if uploaded_file is not None:
                try:
                    self.genai.delete_file(uploaded_file)
                except Exception:
                    pass


def safe_parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(_strip_json_wrappers(s))
    except Exception:
        return {"status": "error", "message": "invalid json from model"}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = GeminiAgent().transcribe(sys.argv[1])
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("Usage: python gemini_agent.py /path/to/file.wav")
