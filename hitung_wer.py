import json
import re
import os
import pandas as pd
from jiwer import wer, cer

# 1. Validasi file log hasil batch pipeline
csv_path = os.path.join("log", "analisis_pipeline.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} tidak ditemukan. Jalankan analisis_pipeline.py terlebih dahulu!")

# 2. Muat Ground Truth JSON
with open(os.path.join("data", "ground_truth.json"), "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# 3. Baca dataset log analisis batch
df = pd.read_csv(csv_path)

list_wer = []
list_cer = []

print("="*50)
print("SISTEM EVALUASI OTOMATIS: PERHITUNGAN WER & CER KORPUS")
print("="*50)

# 4. Iterasi setiap baris log audio yang dievaluasi
for index, row in df.iterrows():
    filename = str(row["File"])
    hypothesis = str(row["Transcript"]) if pd.notna(row["Transcript"]) else ""
    status = str(row["Status"])
    
    if "failed" in status.lower() or not hypothesis:
        continue
        
    # Cari pola "audio" diikuti angka di dalam nama file (misal: 2336_audio1.wav -> audio01)
    match = re.search(r"audio(\d+)", filename)
    if match:
        matched_key = f"audio{match.group(1).zfill(2)}" # Ambil string 'audio01', 'audio12', dst.
        
        if matched_key in ground_truth:
            reference = ground_truth[matched_key]
            
            # Hitung Error Rate menggunakan library jiwer
            error_word = wer(reference, hypothesis)
            error_char = cer(reference, hypothesis)
            
            list_wer.append(error_word)
            list_cer.append(error_char)
            
            print(f"[OK] File: {filename} -> {matched_key} | WER: {error_word:.2%}")
    else:
        print(f"[SKIP] Format nama file tidak sesuai pattern: {filename}")

print("="*50)
# 5. Tampilkan rekapitulasi rata-rata statistik untuk Bab 4 Laporan UAS
if list_wer:
    avg_wer = sum(list_wer) / len(list_wer)
    avg_cer = sum(list_cer) / len(list_cer)
    
    print(f"TOTAL AUDIO SUKSES DIEVALUASI: {len(list_wer)}")
    print(f"RATA-RATA WORD ERROR RATE (WER)  : {avg_wer:.2%}")
    print(f"RATA-RATA CHARACTER ERROR RATE (CER): {avg_cer:.2%}")
    print("="*50)
else:
    print("Tidak ada data audio berstatus sukses yang cocok untuk dihitung.")
    print("Periksa kembali apakah nama file di folder data/audio mengandung kata 'audio1', 'audio2', dst.")
    print("="*50)