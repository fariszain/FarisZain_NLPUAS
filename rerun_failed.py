"""
rerun_failed.py
Jalankan ulang pipeline hanya untuk audio yang LLM-nya terkena fallback
("Maaf, saya tidak bisa memproses permintaan Anda saat ini.")
"""
import os
import re
import time
import pandas as pd

CSV_PATH = "log/analisis_pipeline.csv"
FALLBACK_MSG = "Maaf, saya tidak bisa memproses permintaan Anda saat ini."
CORPUS_DIR = "data/corpus"

# Import modul pipeline
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
from app.utils import normalize_text

def get_corpus_path(filename):
    """Cari file audio di folder corpus."""
    # Coba nama persis dulu
    direct = os.path.join(CORPUS_DIR, filename)
    if os.path.exists(direct):
        return direct
    # Coba tanpa zero-padding (audio01 -> audio1)
    match = re.match(r"^(\d+)_audio0*(\d+)(.wav)$", filename, re.IGNORECASE)
    if match:
        alt = f"{match.group(1)}_audio{match.group(2)}{match.group(3)}"
        alt_path = os.path.join(CORPUS_DIR, alt)
        if os.path.exists(alt_path):
            return alt_path
    return None

def main():
    df = pd.read_csv(CSV_PATH)
    
    # Identifikasi baris fallback
    fallback_mask = df["LLM_Response"].astype(str).str.contains(FALLBACK_MSG, regex=False, na=False)
    total_fallback = fallback_mask.sum()
    print(f"[INFO] Ditemukan {total_fallback} baris dengan LLM fallback. Menjalankan ulang...")
    
    updated = 0
    for idx in df[fallback_mask].index:
        filename = df.at[idx, "File"]
        transcript = df.at[idx, "Transcript"]
        
        if not transcript or str(transcript).startswith("[ERROR]"):
            print(f"  [SKIP] {filename} — transkripsi kosong/error, lewati.")
            continue
        
        print(f"  [RETRY] {filename} ...", end=" ", flush=True)
        try:
            # Panggil LLM ulang
            llm_resp = generate_response(transcript)
            
            if llm_resp and FALLBACK_MSG not in llm_resp:
                # TTS ulang
                tts_ready = normalize_text(llm_resp)
                try:
                    audio_out = transcribe_text_to_speech(tts_ready)
                    df.at[idx, "Audio_Output"] = audio_out
                except Exception as tts_err:
                    print(f"[TTS-WARN] {tts_err}")
                
                df.at[idx, "LLM_Response"] = llm_resp
                df.at[idx, "Status"] = "success"
                updated += 1
                print(f"OK → {llm_resp[:60]}...")
            else:
                print(f"STILL FALLBACK — kuota mungkin masih habis, coba lagi nanti.")
        except Exception as e:
            print(f"ERROR: {e}")
        
        # Simpan progres setiap 10 baris agar tidak hilang jika terhenti
        if updated % 10 == 0 and updated > 0:
            df.to_csv(CSV_PATH, index=False)
            print(f"  [CHECKPOINT] Tersimpan ({updated} diperbarui sejauh ini)")
        
        time.sleep(0.5)  # jeda kecil antar request
    
    # Simpan final
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[SELESAI] {updated}/{total_fallback} baris berhasil diperbarui. CSV disimpan.")

if __name__ == "__main__":
    main()
