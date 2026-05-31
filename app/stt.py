import os
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from .utils import TEMP_DIR, normalize_text, safe_delete
except ImportError:
    from utils import TEMP_DIR, normalize_text, safe_delete

try:
    from processing import preprocess_audio
except ImportError:
    akar_proyek = Path(__file__).resolve().parents[1]
    if str(akar_proyek) not in sys.path:
        sys.path.insert(0, str(akar_proyek))
    from processing import preprocess_audio

LOKASI_BASE = os.path.dirname(os.path.abspath(__file__))
AKAR_PROYEK = os.path.dirname(LOKASI_BASE)

env_whisper_dir = os.getenv("WHISPER_DIR")
if env_whisper_dir:
    DIREKTORI_WHISPER = env_whisper_dir
else:
    candi_app = os.path.join(AKAR_PROYEK, "app", "whisper.cpp")
    candi_models = os.path.join(AKAR_PROYEK, "models", "whisper.cpp")
    DIREKTORI_WHISPER = candi_app if os.path.exists(candi_app) else candi_models

BINER_WHISPER = os.getenv("WHISPER_BINARY", os.path.join(DIREKTORI_WHISPER, "build", "bin", "whisper-cli"))
PATH_MODEL_WHISPER = os.getenv("WHISPER_MODEL_PATH")
if not PATH_MODEL_WHISPER:
    kandidat_model = [
        os.path.join(DIREKTORI_WHISPER, "models", "ggml-base.bin"),
        os.path.join(DIREKTORI_WHISPER, "models", "ggml-base.en.bin"),
        os.path.join(DIREKTORI_WHISPER, "models", "ggml-large-v3-turbo.bin"),
    ]
    for km in kandidat_model:
        if os.path.exists(km):
            PATH_MODEL_WHISPER = km
            break
    else:
        PATH_MODEL_WHISPER = os.path.join(DIREKTORI_WHISPER, "models", "ggml-large-v3-turbo.bin")

TIMEOUT_WHISPER = int(os.getenv("WHISPER_TIMEOUT", "180"))

if os.name == "nt" and not BINER_WHISPER.endswith(".exe"):
    BINER_WHISPER += ".exe"

pipeline_asr_lokal = None


def inisialisasi_pipeline_asr():
    """Menginisialisasi pipeline otomatis speech recognition Hugging Face."""
    global pipeline_asr_lokal
    if pipeline_asr_lokal is None:
        print("[ASR-INFO] Memuat pipeline Whisper Small dari Hugging Face...")
        try:
            import torch
            from transformers import pipeline
            penggunaan_device = 0 if torch.cuda.is_available() else -1
            pipeline_asr_lokal = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=penggunaan_device,
                chunk_length_s=30,
                return_timestamps=False
            )
            print("[ASR-INFO] Pipeline ASR Hugging Face siap digunakan.")
        except Exception as error:
            print(f"[ASR-ERROR] Gagal memuat pipeline ASR: {error}")
            raise error
    return pipeline_asr_lokal


def transcribe_audio_file(audio_path: str) -> str:
    """Mentranskripsikan berkas audio menggunakan whisper.cpp atau fallback Hugging Face."""
    prefix_temp = os.path.join(TEMP_DIR, f"stt_{uuid.uuid4()}")
    path_hasil_txt = f"{prefix_temp}.txt"
    path_prep = None

    try:
        data_prep = preprocess_audio(audio_path, output_dir=TEMP_DIR)
        input_audio = data_prep.output_path if data_prep.status == "success" else audio_path
        path_prep = data_prep.output_path if data_prep.status == "success" else None

        # Cek ketersediaan biner whisper.cpp
        if os.path.exists(BINER_WHISPER) and os.access(BINER_WHISPER, os.X_OK) and os.path.exists(PATH_MODEL_WHISPER):
            cmd = [
                BINER_WHISPER,
                "-m", PATH_MODEL_WHISPER,
                "-l", "auto",
                "-f", input_audio,
                "-otxt",
                "-of", prefix_temp,
            ]
            try:
                hasil = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_WHISPER,
                )
                if hasil.stderr:
                    print(hasil.stderr)
            except subprocess.TimeoutExpired:
                return "[ERROR] Batas waktu transkripsi whisper.cpp habis."
            except subprocess.CalledProcessError as error:
                return f"[ERROR] Eksekusi whisper.cpp gagal: {error.stderr or error}"

            try:
                with open(path_hasil_txt, "r", encoding="utf-8") as f:
                    return normalize_text(f.read())
            except FileNotFoundError:
                return "[ERROR] File hasil transkripsi tidak terbentuk."
        else:
            # Fallback ke pipeline lokal Hugging Face
            print("[ASR-INFO] Biner whisper.cpp tidak ditemukan. Menggunakan Hugging Face Pipeline...")
            try:
                import librosa
                audio_np, sr = librosa.load(input_audio, sr=16000)
                pipe = inisialisasi_pipeline_asr()
                hasil = pipe(audio_np)
                return normalize_text(hasil["text"])
            except Exception as error:
                return f"[ERROR] Gagal memproses fallback ASR: {str(error)}"
    finally:
        safe_delete(path_hasil_txt)
        safe_delete(path_prep)


def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """Mengubah byte data audio mentah menjadi string teks transkripsi."""
    berkas_sementara = os.path.join(TEMP_DIR, f"raw_{uuid.uuid4()}{file_ext}")
    try:
        with open(berkas_sementara, "wb") as f:
            f.write(file_bytes)
        return transcribe_audio_file(berkas_sementara)
    finally:
        safe_delete(berkas_sementara)