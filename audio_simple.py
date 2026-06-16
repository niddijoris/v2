import subprocess
from pathlib import Path
import tempfile


def to_wav(input_path: str) -> str:
    out = str(Path(tempfile.mkdtemp(prefix="audio_") )/ (Path(input_path).stem + ".wav"))
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
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out
