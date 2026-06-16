import os
import tempfile
from typing import Optional, List


def _ensure_pydub():
    try:
        from pydub import AudioSegment, silence
        return AudioSegment, silence
    except Exception:
        return None, None


def convert_to_wav(input_path: str, out_path: Optional[str] = None) -> str:
    """Convert input audio to mono WAV @16kHz. Returns output path.

    Uses `pydub` when available, otherwise falls back to `ffmpeg` CLI.
    """
    AudioSegment, _ = _ensure_pydub()
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    if AudioSegment is not None:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(out_path, format="wav")
        return out_path

    # Fallback to ffmpeg CLI
    import shutil
    from subprocess import CalledProcessError, run

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ImportError("pydub unavailable and `ffmpeg` not found in PATH; install ffmpeg or enable pydub")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ]
    try:
        run(cmd, check=True)
    except CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e}") from e
    return out_path


def chunk_audio_if_large(input_path: str, max_mb: int = 50, chunk_length_ms: int = 10 * 60 * 1000) -> List[str]:
    """If file size > max_mb, split on silence into chunks no longer than chunk_length_ms.
    Returns list of file paths (may be single-item list if no split required).
    """
    size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if size_mb <= max_mb:
        return [input_path]

    AudioSegment, silence = _ensure_pydub()
    out_files: List[str] = []
    # If pydub available, use silence-based splitting
    if AudioSegment is not None:
        audio = AudioSegment.from_wav(input_path)
        parts = silence.split_on_silence(audio, min_silence_len=700, silence_thresh=-40)
        if not parts:
            for i in range(0, len(audio), chunk_length_ms):
                chunk = audio[i : i + chunk_length_ms]
                fd, path = tempfile.mkstemp(suffix=f".part{i}.wav")
                os.close(fd)
                chunk.export(path, format="wav")
                out_files.append(path)
            return out_files

        current = AudioSegment.silent(duration=0)
        idx = 0
        for part in parts:
            if len(current) + len(part) > chunk_length_ms:
                fd, path = tempfile.mkstemp(suffix=f".part{idx}.wav")
                os.close(fd)
                current.export(path, format="wav")
                out_files.append(path)
                idx += 1
                current = part
            else:
                current += part

        if len(current) > 0:
            fd, path = tempfile.mkstemp(suffix=f".part{idx}.wav")
            os.close(fd)
            current.export(path, format="wav")
            out_files.append(path)

        return out_files

    # Fallback: use ffmpeg to create fixed-length segments
    import shutil
    from subprocess import CalledProcessError, run

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ImportError("pydub unavailable and `ffmpeg` not found in PATH; install ffmpeg or enable pydub")

    segment_seconds = int(chunk_length_ms / 1000)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = tempfile.mkdtemp(prefix="segments_")
    pattern = os.path.join(out_dir, f"{base}_%03d.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-c",
        "copy",
        pattern,
    ]
    try:
        run(cmd, check=True)
    except CalledProcessError as e:
        raise RuntimeError(f"ffmpeg segmentation failed: {e}") from e

    # collect generated files
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith(".wav"):
            out_files.append(os.path.join(out_dir, fname))
    return out_files


if __name__ == "__main__":
    print("audio_processor module. Use from other scripts.")
