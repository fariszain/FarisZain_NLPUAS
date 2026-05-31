import os
import re
import shutil
import time
from typing import Dict, Optional

import pandas as pd
from langdetect import detect_langs

from app.llm import generate_response
from app.stt import transcribe_speech_to_text
from app.tts import transcribe_text_to_speech
from app.utils import normalize_text
from processing import preprocess_audio

# IMPORT fungsi pembersih data DAN kamus kunci jawaban langsung dari berkas bersihkan_data punyamu
from bersihkan_data import saring_dan_ambil_data_bersih, REFERENCE_TRANSCRIPTS


# ========================================================
# METRIK MATEMATIKA (LOGIKA EVALUASI WER & CER)
# ========================================================

def _levenshtein(s1: list, s2: list) -> int:
    """Menghitung jarak edit Levenshtein antara dua list."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _clean_for_wer(text: str) -> str:
    """Standardisasi teks dasar agar penghitungan WER/CER adil."""
    if not text:
        return ""
    # Hapus harakat bahasa Arab jika ada
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    # Hapus tanda baca umum dan jadikan spasi tunggal
    text = re.sub(r'[.,!?؛،؟"\'\-_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def calculate_wer(ref: str, hyp: str) -> float:
    """Menghitung Word Error Rate."""
    r = _clean_for_wer(ref).split()
    h = _clean_for_wer(hyp).split()
    return round(min(1.0, _levenshtein(r, h) / len(r)), 4) if r else (1.0 if h else 0.0)


def calculate_cer(ref: str, hyp: str) -> float:
    """Menghitung Character Error Rate (Tanpa Spasi)."""
    r = list(_clean_for_wer(ref).replace(" ", ""))
    h = list(_clean_for_wer(hyp).replace(" ", ""))
    return round(min(1.0, _levenshtein(r, h) / len(r)), 4) if r else (1.0 if h else 0.0)


# ========================================================
# ANALISIS BAHASA
# ========================================================

def _analyze_code_switching(text: str) -> Dict[str, float]:
    """Menghitung rasio pencampuran bahasa secara spesifik untuk masing-masing 
    bahasa: Inggris (EN), Indonesia (ID), dan Arab (AR) menggunakan langdetect."""
    if not text or not re.sub(r"[^\w\s]", "", text).strip():
        return {"EN": 0.0, "ID": 0.0, "AR": 0.0}

    try:
        text_clean = re.sub(r"[^\w\s]", "", text)
        predictions = detect_langs(text_clean)
        lang_ratios = {"EN": 0.0, "ID": 0.0, "AR": 0.0}
        
        for pred in predictions:
            if pred.lang == 'en':
                lang_ratios["EN"] = round(pred.prob, 3)
            elif pred.lang in ['id', 'ms']:
                lang_ratios["ID"] = round(lang_ratios["ID"] + pred.prob, 3)
            elif pred.lang == 'ar':
                lang_ratios["AR"] = round(pred.prob, 3)
                
        return lang_ratios
    except Exception:
        return {"EN": 0.0, "ID": 1.0, "AR": 0.0}


def jalankan_uji_korpus(folder_corpus_audio: str, student_prefix: Optional[str] = None):
    """Jalankan evaluasi batch pada folder audio dengan integrasi bersihkan_data, WER/CER & Checkpoint."""
    
    log_dir = os.path.join("log")
    os.makedirs(log_dir, exist_ok=True)
    out_csv = os.path.join(log_dir, "analisis_pipeline.csv")

    print(f"[PRE-PROCESS] Memanggil bersihkan_data untuk menyaring folder {folder_corpus_audio}...")
    semua_metadata = saring_dan_ambil_data_bersih(folder_corpus_audio)
    
    files_to_process = []
    for item in semua_metadata:
        if not item["is_valid"]:
            continue
        if student_prefix and not item["filename_baru"].startswith(student_prefix):
            continue
        files_to_process.append(item)

    print(f"[INFO] Total audio unik & valid yang akan diproses pada batch ini: {len(files_to_process)} file.")

    if not files_to_process:
        print("[INFO] Tidak ada file yang memenuhi kriteria untuk diproses.")
        return

    # LOAD CHECKPOINT (Auto-Resume)
    progres_sebelumnya = {}
    if os.path.exists(out_csv):
        try:
            df_old = pd.read_csv(out_csv)
            for _, row in df_old.iterrows():
                if str(row.get("Status", "")).strip().lower() == "success":
                    progres_sebelumnya[row["File"]] = {
                        "Transcript": row.get("Transcript", ""),
                        "Normalized": row.get("Normalized", ""),
                        "WER": row.get("WER", 1.0),
                        "CER": row.get("CER", 1.0),
                        "Ratios": row.get("Ratios", "EN:0.0, ID:0.0, AR:0.0"),
                        "LLM_Response": row.get("LLM_Response", ""),
                        "Audio_Output": row.get("Audio_Output", ""),
                        "Latency_s": row.get("Latency_s", 0.0),
                        "Status": "success"
                    }
            print(f"[CHECKPOINT] Menemukan log lama. {len(progres_sebelumnya)} audio sudah sukses dan akan dilewati.")
        except Exception as e:
            print(f"[CHECKPOINT WARNING] Gagal membaca checkpoint lama: {e}")

    hasil_analisis = []
    temp_dir = os.path.join("temp", "pipeline")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    output_audio_dir = os.path.join("data", "output_audio")
    os.makedirs(output_audio_dir, exist_ok=True)

    try:
        for item in files_to_process:
            path_asli = item["path_asli"]
            filename_lama = item["filename_lama"]
            filename_baru = item["filename_baru"]  # Format rapi ber-ekstensi audio (e.g., 2336_audio01.wav)
            key_jawaban = item["utterance_id"]     # Berisi string "audio01", "audio02" dll.
            base_filename = os.path.splitext(filename_baru)[0]

            if filename_baru in progres_sebelumnya:
                print(f"Skipping (Sudah Sukses): {filename_baru}")
                old_data = progres_sebelumnya[filename_baru]
                hasil_analisis.append({
                    "file": filename_baru,
                    "transcript": old_data["Transcript"],
                    "normalized": old_data["Normalized"],
                    "wer": old_data["WER"],
                    "cer": old_data["CER"],
                    "ratios_str": old_data["Ratios"],
                    "llm_response": old_data["LLM_Response"],
                    "audio_output": old_data["Audio_Output"],
                    "latency_s": old_data["Latency_s"],
                    "status": "success",
                })
                continue

            print(f"Memproses Audio: {filename_lama} ➔ {filename_baru}")
            start_time = time.time()

            try:
                processed = preprocess_audio(path_asli, output_dir=temp_dir)
                if processed.status != "success":
                    raise RuntimeError(processed.message or "preprocessing gagal")

                with open(processed.output_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                try:
                    os.remove(processed.output_path)
                except OSError:
                    pass

                # 1. Komponen Speech-to-Text (STT)
                transcript = transcribe_speech_to_text(audio_bytes)
                normalized = normalize_text(transcript)
                
                # 2. EVALUASI AKURASI INSTAN (Menghubungkan langsung ke Kamus bersihkan_data)
                teks_referensi = REFERENCE_TRANSCRIPTS.get(key_jawaban, "")
                if teks_referensi:
                    skor_wer = calculate_wer(teks_referensi, normalized)
                    skor_cer = calculate_cer(teks_referensi, normalized)
                else:
                    skor_wer, skor_cer = 1.0, 1.0 # default nilai jika key salah

                # 3. Hitung Rasio Bahasa
                ratios = _analyze_code_switching(transcript)
                ratios_str = ", ".join([f"{k}:{v}" for k, v in ratios.items()]) if ratios else "EN:0.0, ID:0.0, AR:0.0"
                
                # 4. Panggil Generative LLM
                raw_llm_resp = generate_response(normalized)
                llm_resp = normalize_text(raw_llm_resp)
                
                # 5. Convert Response ke Audio (TTS)
                audio_output_path = None
                if llm_resp and llm_resp.strip():
                    try:
                        audio_output_path = transcribe_text_to_speech(llm_resp)
                        final_audio_path = os.path.join(output_audio_dir, f"{base_filename}_response.wav")
                        
                        shutil.move(audio_output_path, final_audio_path)
                        audio_output_path = final_audio_path
                    except Exception as tts_exc:
                        print(f"Warning: TTS gagal untuk {filename_baru}: {tts_exc}")
                        audio_output_path = None

                elapsed = round(time.time() - start_time, 2)
                hasil_analisis.append({
                    "file": filename_baru,
                    "transcript": transcript,
                    "normalized": normalized,
                    "wer": skor_wer,
                    "cer": skor_cer,
                    "ratios_str": ratios_str,
                    "llm_response": llm_resp,
                    "audio_output": audio_output_path or "",
                    "latency_s": elapsed,
                    "status": "success",
                })
                print(f" ➔ [BERHASIL] WER: {skor_wer} | CER: {skor_cer}")
                
            except Exception as exc:
                elapsed = round(time.time() - start_time, 2)
                print(f"Gagal memproses {filename_lama}: {exc}")
                hasil_analisis.append({
                    "file": filename_baru,
                    "transcript": "",
                    "normalized": "",
                    "wer": 1.0,
                    "cer": 1.0,
                    "ratios_str": "EN:0.0, ID:0.0, AR:0.0",
                    "llm_response": "",
                    "audio_output": "",
                    "status": f"failed: {exc}",
                    "latency_s": elapsed,
                })

            # Backup progress ke CSV setiap kali selesai 1 audio (Real-time saving)
            _save_progress_to_csv(hasil_analisis, out_csv)
            time.sleep(6)

        print(f"\n[SUKSES] Semua proses evaluasi korpus selesai! Laporan akhir: {out_csv}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _save_progress_to_csv(hasil_analisis, out_csv):
    """Fungsi pembantu untuk menulis log langsung ke CSV secara real-time."""
    rows = []
    for item in hasil_analisis:
        rows.append({
            "File": item["file"],
            "Transcript": item["transcript"],
            "Normalized": item["normalized"],
            "WER": item["wer"],
            "CER": item["cer"],
            "Ratios": item["ratios_str"],
            "LLM_Response": item["llm_response"],
            "Audio_Output": item["audio_output"],
            "Latency_s": item["latency_s"],
            "Status": item["status"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    try:
        if os.path.exists(os.path.join("data", "corpus")):
            folder = os.path.join("data", "corpus")
        else:
            folder = os.path.join("data", "audio")
        # Jalankan langsung dengan parameter folder audio tanpa filter prefix NPM agar memproses seluruh korpus
        jalankan_uji_korpus(folder, student_prefix=None)
    except Exception as exc:
        print(f"Error: {exc}")