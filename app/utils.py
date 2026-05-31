import os
import re
from typing import Optional
from num2words import num2words

# Inisialisasi direktori kerja
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(DIR_BASE, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def ubah_angka_ke_teks(teks_input: str) -> str:
    """
    Mengubah deret angka numerik di dalam string menjadi padanannya dalam kata terbilang Bahasa Indonesia.
    Contoh: "12" -> "dua belas"
    """
    pola_angka = re.compile(r'\d+')
    
    def pengganti_kata(match):
        teks_angka = match.group(0)
        try:
            return num2words(int(teks_angka), lang='id')
        except Exception:
            return teks_angka

    return pola_angka.sub(pengganti_kata, teks_input)


def normalize_text(text: str) -> str:
    """Melakukan normalisasi teks sebelum diumpankan ke model LLM & modul TTS."""
    if not text:
        return ""

    teks_bersih = text.strip()
    
    # Ubah format angka
    teks_bersih = ubah_angka_ke_teks(teks_bersih)
    
    # Perapian spasi dan tanda baca
    teks_bersih = re.sub(r"\s+", " ", teks_bersih)
    teks_bersih = re.sub(r"\s+([,.!?;:])", r"\1", teks_bersih)
    teks_bersih = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", teks_bersih)
    return teks_bersih


def safe_delete(path: Optional[str]) -> None:
    """Menghapus berkas temporary secara aman tanpa memicu crash aplikasi."""
    if not path:
        return
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception as error:
        print(f"[WARN-UTILS] Gagal menghapus berkas {path}: {error}")


def get_file_ext(filename: str, default: str = ".wav") -> str:
    """Mengembalikan ekstensi berkas dalam format huruf kecil."""
    nama_ekstensi = os.path.splitext(filename or "")[1].lower()
    return nama_ekstensi if nama_ekstensi else default