import json
import os
import re
import time
from datetime import date
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from .utils import normalize_text
except ImportError:
    from utils import normalize_text

LOKASI_BASE = os.path.dirname(os.path.abspath(__file__))
DIREKTORI_PROYEK = os.path.dirname(LOKASI_BASE)
PATH_DOTENV = os.path.join(DIREKTORI_PROYEK, ".env")
load_dotenv(PATH_DOTENV, override=True)

DIREKTORI_KONTROL = os.path.join(DIREKTORI_PROYEK, "storage")
os.makedirs(DIREKTORI_KONTROL, exist_ok=True)
FILE_RIWAYAT_CHAT = os.path.join(DIREKTORI_KONTROL, "chat_history.json")
FILE_BATAS_KONTROL = os.path.join(DIREKTORI_KONTROL, "rate_state.json")

NAMA_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
BATAS_RPM = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
BATAS_RPD = int(os.getenv("GEMINI_RPD_LIMIT", "4500"))
BATAS_WAKTU_REQ = int(os.getenv("GEMINI_TIMEOUT", "60"))
MAKSIMAL_PERCOBAAN = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# Pengolahan daftar kunci API
DAFTAR_KUNCI_API = []
kunci_raw = os.getenv("GEMINI_API_KEYS", "")
if kunci_raw:
    DAFTAR_KUNCI_API = [k.strip() for k in kunci_raw.split(",") if k.strip()]
if not DAFTAR_KUNCI_API:
    kunci_tunggal = os.getenv("GEMINI_API_KEY")
    if kunci_tunggal:
        DAFTAR_KUNCI_API = [kunci_tunggal]

if not DAFTAR_KUNCI_API:
    raise RuntimeError("Kunci API Gemini tidak ditemukan di .env!")

indeks_kunci_sekarang = 0


def dapatkan_klien_aktif() -> genai.Client:
    """Mengembalikan objek klien GenAI dengan API key yang aktif."""
    global indeks_kunci_sekarang
    kunci_aktif = DAFTAR_KUNCI_API[indeks_kunci_sekarang % len(DAFTAR_KUNCI_API)]
    return genai.Client(
        api_key=kunci_aktif,
        http_options=types.HttpOptions(timeout=BATAS_WAKTU_REQ * 1000)
    )


INSTRUKSI_SISTEM = """
You are a direct conversational virtual assistant.
Task: Answer the user's question immediately.

STRICT RULES:
1. Output ONLY the final conversational answer in Indonesian.
2. Absolutely NO explanations, NO drafts, NO thoughts, NO format logs, NO self-corrections, and NO analysis text.
3. Maximum 2-3 sentences.
""".strip()

konfigurasi_chat = types.GenerateContentConfig(
    system_instruction=INSTRUKSI_SISTEM,
    temperature=0.3,
    max_output_tokens=1024,
)


def baca_data_json(jalur_file: str, nilai_default: Any) -> Any:
    """Membaca berkas JSON dengan aman."""
    if not os.path.exists(jalur_file) or os.path.getsize(jalur_file) == 0:
        return nilai_default
    try:
        with open(jalur_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return nilai_default


def tulis_data_json(jalur_file: str, data_simpan: Any) -> None:
    """Menulis data ke berkas JSON."""
    with open(jalur_file, "w", encoding="utf-8") as f:
        json.dump(data_simpan, f, ensure_ascii=False, indent=2)


def batasi_rate_limit_lokal() -> None:
    """Fungsi pencegah request berlebih agar tidak melampaui RPM/RPD."""
    waktu_sekarang = time.time()
    tanggal_hari_ini = date.today().isoformat()
    status_limit = baca_data_json(FILE_BATAS_KONTROL, {"date": tanggal_hari_ini, "daily_count": 0, "timestamps": []})

    if status_limit.get("date") != tanggal_hari_ini:
        status_limit = {"date": tanggal_hari_ini, "daily_count": 0, "timestamps": []}

    if status_limit.get("daily_count", 0) >= BATAS_RPD:
        raise RuntimeError(f"Batas harian RPD tercapai ({BATAS_RPD}).")

    list_timestamp = [t for t in status_limit.get("timestamps", []) if waktu_sekarang - float(t) < 60]
    if len(list_timestamp) >= BATAS_RPM:
        durasi_tunggu = 60 - (waktu_sekarang - min(list_timestamp)) + 1
        print(f"[INFO-LLM] Batas RPM tercapai. Menunggu {durasi_tunggu:.1f} detik...")
        time.sleep(max(1, durasi_tunggu))
        waktu_sekarang = time.time()
        list_timestamp = [t for t in list_timestamp if waktu_sekarang - float(t) < 60]

    list_timestamp.append(waktu_sekarang)
    status_limit["timestamps"] = list_timestamp
    status_limit["daily_count"] = status_limit.get("daily_count", 0) + 1
    tulis_data_json(FILE_BATAS_KONTROL, status_limit)


def generate_response(prompt: str) -> str:
    """Mengirim prompt teks ke Gemini API dan merapikan output responsnya."""
    global indeks_kunci_sekarang
    prompt_bersih = normalize_text(prompt)
    if not prompt_bersih:
        return "Maaf, pesan input tidak terdeteksi dengan jelas."

    kesalahan_terakhir = None
    for percobaan in range(1, len(DAFTAR_KUNCI_API) + 1):
        try:
            batasi_rate_limit_lokal()
            klien = dapatkan_klien_aktif()
            print(f"[LLM-DEBUG] Kunci #{indeks_kunci_sekarang + 1} | Kirim ke: {NAMA_MODEL}...")
            
            respons_raw = klien.models.generate_content(
                model=NAMA_MODEL,
                contents=prompt_bersih,
                config=konfigurasi_chat
            )
            
            teks_jawaban = None
            if hasattr(respons_raw, 'text') and respons_raw.text:
                teks_jawaban = respons_raw.text
            elif hasattr(respons_raw, 'candidates') and respons_raw.candidates:
                try:
                    kandidat = respons_raw.candidates[0]
                    if hasattr(kandidat, 'content') and hasattr(kandidat.content, 'parts'):
                        if kandidat.content.parts:
                            part_data = kandidat.content.parts[0]
                            if hasattr(part_data, 'text') and part_data.text:
                                teks_jawaban = part_data.text
                except Exception as exc_candidate:
                    print(f"[LLM-WARN] Gagal parse kandidat: {exc_candidate}")
            
            if teks_jawaban and teks_jawaban.strip():
                # Membersihkan karakter khusus markdown dan tanda baca berlebih
                teks_jawaban = teks_jawaban.replace("*", "").replace("#", "").replace('"', '').replace('(', '').replace(')', '')
                baris_data = [b.strip() for b in teks_jawaban.split('\n') if b.strip()]
                teks_final = ""
                for baris in reversed(baris_data):
                    b_lower = baris.lower()
                    if ":" in baris or any(x in b_lower for x in [
                        'user input', 'intent', 'goal', 'constraints', 'language', 
                        'draft', 'refining', 'self-correction', 'user asks', 
                        'option', 'direct?', 'no analysis', 'polite/clear', 'context', 'translation'
                    ]):
                        continue
                    teks_final = baris
                    break
                
                if not teks_final and baris_data:
                    teks_final = baris_data[-1]
                teks_jawaban = teks_final
                
                if teks_jawaban:
                    idx_punc = max(teks_jawaban.rfind('.'), teks_jawaban.rfind('?'), teks_jawaban.rfind('!'))
                    if idx_punc != -1:
                        teks_jawaban = teks_jawaban[:idx_punc + 1].strip()

                return normalize_text(teks_jawaban)
            else:
                raise RuntimeError("Respons dari model kosong.")
                
        except Exception as error:
            kesalahan_terakhir = error
            pesan_error = str(error).lower()
            is_limit = any(k in pesan_error or k in type(error).__name__.lower() for k in ["429", "resource_exhausted", "quota", "rate"])
            
            if is_limit:
                indeks_lama = indeks_kunci_sekarang
                indeks_kunci_sekarang = (indeks_kunci_sekarang + 1) % len(DAFTAR_KUNCI_API)
                print(f"[LLM-ROTATE] Kunci #{indeks_lama + 1} limit. Beralih ke #{indeks_kunci_sekarang + 1}...")
                time.sleep(1)
                continue
            else:
                print(f"[LLM-WARN] Percobaan ke-{percobaan} gagal: {error}")
                indeks_kunci_sekarang = (indeks_kunci_sekarang + 1) % len(DAFTAR_KUNCI_API)
                time.sleep(1)

    print(f"[LLM-ERROR] Semua kunci API habis/gagal. Detail: {kesalahan_terakhir}")
    return "Maaf, layanan chatbot sedang mengalami kepadatan trafik. Silakan coba kembali."