import asyncio
import os
import subprocess
import sys
import uuid
import re

try:
    from .utils import TEMP_DIR, normalize_text
except ImportError:
    from utils import TEMP_DIR, normalize_text

# Pengaturan direktori Coqui
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COQUI_DIR = os.getenv("COQUI_DIR")
if not COQUI_DIR:
    cand1 = os.path.join(BASE_DIR, "coqui_utils")
    cand2 = os.path.join(BASE_DIR, "coqui_tts")
    COQUI_DIR = cand1 if os.path.exists(cand1) else cand2

COQUI_MODEL_PATH = os.getenv("COQUI_MODEL_PATH", os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth"))
COQUI_CONFIG_PATH = os.getenv("COQUI_CONFIG_PATH", os.path.join(COQUI_DIR, "config.json"))
COQUI_SPEAKER = os.getenv("COQUI_SPEAKER", "wibowo")
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "180"))
TTS_BINARY_DEFAULT = os.path.join(os.path.dirname(sys.executable), "tts")
TTS_BINARY = os.getenv("TTS_BINARY", TTS_BINARY_DEFAULT)

# Single best voice for natural Indonesian speech
BEST_TTS_VOICE = "id-ID-ArdiNeural"

# Pemetaan fonetis kustom untuk memperbaiki lafal kata
KAMUS_PELAFALAN = {
    r"\brombongan\b": "rombonggan",
    r"\bproses\b": "peroses",
    r"\bumrah\b": "umroh",
    r"\bflight\b": "flait",
    r"\bschedule\b": "skedul",
    r"\bbooking\b": "buking",
    r"\btransport\b": "trans-port",
    r"\btravel\b": "trevel",
    r"\bsimple\b": "simpel",
    r"\binclude\b": "inklud",
    r"\btomorrow\b": "tumoro",
    r"\bfasting\b": "fes-ting",
    r"\bhajj\b": "haj",
    r"\bapply\b": "eplay",
    r"\bexplain\b": "eks-plein",
}


def bersihkan_untuk_tts(teks: str) -> str:
    """Mengoreksi pelafalan teks agar terdengar sempurna dan tidak cadel/sumbing."""
    if not teks:
        return ""
    t_bersih = teks.lower()
    for pola, pengganti in KAMUS_PELAFALAN.items():
        t_bersih = re.sub(pola, pengganti, t_bersih)
    return t_bersih


def cek_modul_edge() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def cek_model_coqui() -> bool:
    return (
        os.path.exists(COQUI_MODEL_PATH)
        and os.path.exists(COQUI_CONFIG_PATH)
        and os.path.exists(TTS_BINARY)
    )


async def jalankan_edge_tts_async(teks: str, file_output: str) -> None:
    import edge_tts
    com = edge_tts.Communicate(teks, BEST_TTS_VOICE)
    temp_mp3 = file_output.replace(".wav", ".mp3")
    await com.save(temp_mp3)
    
    # Konversi MP3 ke WAV agar sesuai format standard audio
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "22050", "-ac", "1", file_output],
            check=True, capture_output=True
        )
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
    except Exception:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_mp3)
            audio = audio.set_frame_rate(22050).set_channels(1)
            audio.export(file_output, format="wav")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception:
            os.rename(temp_mp3, file_output.replace(".wav", ".mp3"))


def sintesis_edge_tts(teks: str) -> str:
    path_wav = os.path.join(TEMP_DIR, f"voice_{uuid.uuid4()}.wav")
    asyncio.run(jalankan_edge_tts_async(teks, path_wav))
    return path_wav


def sintesis_coqui_tts(teks: str) -> str:
    path_wav = os.path.join(TEMP_DIR, f"coqui_{uuid.uuid4()}.wav")
    cmd = [
        TTS_BINARY,
        "--text", teks,
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--out_path", path_wav,
    ]
    if COQUI_SPEAKER:
        cmd.extend(["--speaker_idx", COQUI_SPEAKER])

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=TTS_TIMEOUT, cwd=COQUI_DIR
        )
        return path_wav
    except Exception as err:
        raise RuntimeError(f"Gagal menjalankan sintesis Coqui: {err}")


def transcribe_text_to_speech(text: str) -> str:
    """Mengubah teks ke audio ucapan dengan 1 model terbaik & natural."""
    teks_mentah = normalize_text(text)
    if not teks_mentah:
        raise ValueError("Teks kosong.")
    
    # Koreksi pelafalan
    teks_siap = bersihkan_untuk_tts(teks_mentah)
    print(f"[TTS-PROCESS] Menyintesis teks: '{teks_siap}'")
    
    # Edge TTS (Terbaik & Sangat Natural)
    if cek_modul_edge():
        try:
            return sintesis_edge_tts(teks_siap)
        except Exception as error:
            print(f"[TTS-WARNING] Gagal menggunakan Edge TTS: {error}. Beralih ke Coqui...")
    
    # Fallback Coqui
    if cek_model_coqui():
        return sintesis_coqui_tts(teks_siap)
        
    raise RuntimeError("Tidak ada mesin TTS yang tersedia.")