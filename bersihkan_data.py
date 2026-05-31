import os
import re
from pathlib import Path

# Definisikan dictionary referensi internal untuk validasi kecocokan ID Utterance
REFERENCE_TRANSCRIPTS = {
    "audio01": "Aku mau book flight ke Jeddah minggu depan, bisa bantu schedule?",
    "audio02": "Aku butuh travel umrah simple tapi include Madinah visit",
    "audio03": "Can you help aku arrange transport dari Jeddah ke Madinah tomorrow",
    "audio04": "Explain step by step cara apply visa Saudi dengan benar",
    "audio05": "Ya akhi, uridu book flight ila Jeddah al-usbu'al qadim. Hal bisa bantu ajida afdhal schedule wa rihlatan mubashirah?",
    "audio06": "Uridu arrange transport min Jeddah ila Madinah ghadan",
    "audio07": "Book flight ke Jeddah lalu lanjut ke Madinah, schedule terbaik kapan",
    "audio08": "Uridu schedule trip dari Jeddah ke Makkah besok pagi",
    "audio09": "Mumkin book transport min makkah ila madinah untuk besok?",
    "audio10": "Apa perbedaan umrah dan hajj secara detail dalam Islam",
    "audio11": "Kenapa fasting di Ramadhan itu wajib bagi Muslim",
    "audio12": "Bagaimana proses visa Saudi untuk umrah dari Indonesia sekarang",
    "audio13": "Jelaskan step by step cara booking flight ke Jeddah secara online",
    "audio14": "How to prepare dokumen umrah dari Indonesia dengan benar",
    "audio15": "Tolong buat checklist persiapan umrah termasuk barang wajib dibawa",
    "audio16": "Guide aku cara pilih hotel di Makkah dekat Haram dengan budget terbatas",
    "audio17": "Menurut kamu belajar bahasa Arab itu susah gak untuk pemula",
    "audio18": "I feel overwhelmed dengan persiapan umrah, ada tips sederhana?",
    "audio19": "Ahyanan saya bingung mulai dari mana untuk umrah",
    "audio20": "Translate ke English: aku mau pergi ke Makkah minggu depan"
}


def normalize_and_validate_filename(filename: str) -> tuple[bool, str, str, str, str]:
    """
    Fuzzy parsing nama file audio dan validasi keaslian format tugas.
    Menghapus extension .m4a gantung, string (1)/(2) duplikat download,
    dan mengembalikan nama standar NNNN_NN.wav
    """
    norm = filename.lower()
    norm = norm.replace(".m4a", "")
    norm = re.sub(r'\(\d+\)', '', norm)  # Menghapus pola nama gantung (1), (2), dst.
    norm = norm.strip()
    
    # regex pencocokan: 4 digit (NPM) + opsional kata '_audio' + 1 atau 2 digit (ID Utterance)
    match = re.search(r'^(\d{4})_?(?:audio)?(\d{1,2})\.wav$', norm)
    if not match:
        return False, "", "", norm, f"Gagal dinormalisasi (Format bukan NNNN_audioNN.wav), terdeteksi: '{filename}'"

    student_id = match.group(1)
    utterance_id = match.group(2).zfill(2)  # Menyeragamkan padding angka tunggal menjadi 2 digit (e.g. audio1 -> audio01)
    
    # Tetap kembalikan format asli audioXX agar konsisten dengan keinginanmu
    normalized_filename = f"{student_id}_audio{utterance_id}.wav"
    key_jawaban = f"audio{utterance_id}"

    if key_jawaban not in REFERENCE_TRANSCRIPTS:
        return (
            False, student_id, key_jawaban, normalized_filename,
            f"Utterance ID '{key_jawaban}' di luar batas kamus korpus (audio01–audio20)"
        )

    return True, student_id, key_jawaban, normalized_filename, ""


def saring_dan_ambil_data_bersih(corpus_dir: str) -> list[dict]:
    """
    Memindai direktori audio, melakukan pembersihan nama file, 
    dan secara otomatis menyeleksi file dengan ukuran kapasitas (size) terbesar 
    paling lengkap data suaranya jika terdeteksi adanya duplikasi ID audio yang sama.
    """
    if not os.path.exists(corpus_dir):
        print(f"[ERROR] Folder '{corpus_dir}' tidak ditemukan!")
        return []

    wav_files = []
    for root, _, files in os.walk(corpus_dir):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    temp_dedup_dict = {}

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        is_valid, sid, uid, norm_fname, reason = normalize_and_validate_filename(fname)
        file_size = os.path.getsize(wav_path)

        meta_file = {
            "path_asli": wav_path,
            "filename_lama": fname,
            "filename_baru": norm_fname,
            "is_valid": is_valid,
            "student_id": sid,
            "utterance_id": uid, # Berisi "audio01", "audio02", dst.
            "size_bytes": file_size,
            "reason": reason
        }

        if norm_fname in temp_dedup_dict:
            if temp_dedup_dict[norm_fname]["size_bytes"] < file_size:
                temp_dedup_dict[norm_fname] = meta_file
        else:
            temp_dedup_dict[norm_fname] = meta_file

    hasil_bersih = list(temp_dedup_dict.values())
    hasil_bersih.sort(key=lambda x: (x["student_id"], x["utterance_id"]))
    return hasil_bersih


if __name__ == "__main__":
    if os.path.exists(os.path.join("data", "corpus")):
        FOLDER_TARGET = os.path.join("data", "corpus")
    else:
        FOLDER_TARGET = os.path.join("data", "audio")
    print(f"=== MEMULAI PENYARINGAN DATA KORPUS (Folder: {FOLDER_TARGET}) ===")
    data_siap_pakai = saring_dan_ambil_data_bersih(FOLDER_TARGET)
    print(f"\n[INFO] Total data unik & bersih yang siap dimasukkan ke pipeline kamu: {len(data_siap_pakai)} file.")