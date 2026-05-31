import base64
import os
import re
import traceback
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

try:
    from .llm import generate_response
    from .stt import transcribe_audio_file
    from .tts import transcribe_text_to_speech
    from .utils import TEMP_DIR, get_file_ext, normalize_text, safe_delete
except ImportError:
    from llm import generate_response
    from stt import transcribe_audio_file
    from tts import transcribe_text_to_speech
    from utils import TEMP_DIR, get_file_ext, normalize_text, safe_delete

app = FastAPI(title="Sistem API Asisten Percakapan Multibahasa", version="1.1.0")

# Konfigurasi CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def deteksi_proporsi_bahasa(teks_input: str):
    """
    Menganalisis proporsi kata berdasarkan bahasa (Indonesia, Inggris, Arab)
    untuk kebutuhan diagnosis dan visualisasi pada dashboard.
    """
    if not teks_input:
        return {}, {}

    daftar_token = re.findall(r"[A-Za-z]+|[ء-ي]+", teks_input.lower())
    if not daftar_token:
        return {}, {}

    kosakata_indonesia = {
        "aku", "saya", "kamu", "anda", "kita", "mereka", "mau", "ingin", "boleh",
        "bisa", "tolong", "bantu", "mohon", "jadwal", "pesawat", "tiket", "penerbangan",
        "tanggal", "jam", "hari", "bulan", "tahun", "kapan", "siapa", "bagaimana", "dimana",
        "kemana", "darimana", "dari", "ke", "di", "dan", "atau", "yang", "untuk", "dengan",
        "sudah", "belum", "tidak", "jangan", "coba", "minta", "sampai", "pagi", "siang",
        "malam", "besok", "lusa", "sekarang", "berapa", "satu", "dua", "tiga", "empat",
        "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "coba", "gimana",
        "kalo", "tau", "nggak", "gak", "nih", "dong", "deh", "sih", "banget", "bener",
        "kalo", "dulu", "nanti", "apa", "aja", "ya", "lah", "loh", "kan", "pun",
        "fyi", "gue", "lu", "lo", "cuy", "abis", "emang", "tuh", "gitu", "gini",
    }
    
    gaul_indonesia = {
        "gue", "lu", "lo", "cuy", "nih", "dong", "deh", "sih", "banget", "abis", "gitu",
        "gini", "tuh", "emang", "ya", "lah", "loh", "kan", "ngeh", "bentar", "baca",
    }
    
    kosakata_inggris = {
        "i", "you", "we", "they", "he", "she", "it", "am", "is", "are", "was", "were",
        "be", "been", "being", "want", "need", "can", "could", "would", "should", "will",
        "do", "does", "did", "have", "has", "had", "to", "for", "from", "with", "this",
        "that", "these", "those", "please", "book", "flight", "schedule", "jeddah", "mingo",
        "japan", "airport", "hotel", "morning", "afternoon", "night", "tomorrow", "sunday",
        "january", "google", "flights", "skyscanner", "help", "thanks", "thank", "yes", "no",
        "go", "check", "directly", "real", "time", "can", "you", "me", "my", "your", "our",
        "hello", "hi", "hey", "okay", "ok", "please", "directly", "book", "route", "ticket",
    }
    
    gaul_inggris = {
        "yo", "dude", "bro", "sup", "pls", "lol", "idk", "u", "ur", "gonna", "wanna",
        "kinda", "ain", "aint", "nah", "hmm", "okay", "ok",
    }
    
    kosakata_arab = {
        "سلام", "عليكم", "وعليكم", "صل", "سلم", "سلام", "الله", "الحمد", "لله",
        "الحمد لله", "سبحان", "تعالى", "يا", "ربي", "ربنا", "تبارك", "الله",
        "مساء", "صباح", "مرحبا", "كيف", "الحال", "اليوم", "احsen", "شكرا", "برك",
        "شكرا", "الحمدلله", "اللهم", "تسلمي", "فيه", "عربي", "عربيه",
    }
    
    gaul_arab = {
        "assalamualaikum", "alhamdulillah", "inshaallah", "masyaallah", "subhanallah",
        "habibi", "habibti", "wallahi", "jazakallah", "jazakillah", "salam", "salamualaikum",
    }

    skor_ind = 0
    skor_eng = 0
    skor_ara = 0

    for kata in daftar_token:
        # Deteksi huruf Arab asli
        if re.fullmatch(r"[ء-ي]+", kata):
            skor_ara += 1
            continue

        if kata in kosakata_arab or kata in gaul_arab:
            skor_ara += 1
            continue

        if kata in kosakata_inggris or kata in gaul_inggris:
            skor_eng += 1
            continue

        if kata in kosakata_indonesia or kata in gaul_indonesia or kata.endswith(("nya", "lah", "kan", "kah", "pun", "ku", "mu", "deh", "sih", "dong", "nih")):
            skor_ind += 1
            continue

        # Deteksi akhiran bahasa Inggris umum
        if kata.endswith(("ing", "tion", "ment", "ly", "ize", "ise", "ed", "er")):
            skor_eng += 1
            continue

        skor_ind += 1

    total_kata = len(daftar_token)
    persentase = {}
    if skor_ara:
        persentase["AR"] = round(skor_ara / total_kata, 3)
    if skor_eng:
        persentase["EN"] = round(skor_eng / total_kata, 3)
    if skor_ind:
        persentase["IND"] = round(skor_ind / total_kata, 3)

    # Penyederhanaan jika salah satu bahasa sangat dominan
    if "EN" in persentase and persentase["EN"] >= 0.8 and persentase.get("IND", 0) < 0.2:
        persentase = {"EN": round(persentase["EN"], 3)}
    elif "AR" in persentase and persentase["AR"] >= 0.8 and persentase.get("IND", 0) < 0.2:
        persentase = {"AR": round(persentase["AR"], 3)}

    tag_label = {k: f"{v:.0%}" for k, v in persentase.items()}
    return tag_label, persentase


def susun_respons_json(
    transkripsi: str,
    jawaban_llm: str,
    teks_normalisasi: str,
    pilihan_mode: str,
    lokasi_suara: str,
    back_task: BackgroundTask | None = None,
):
    tag_label, persentase = deteksi_proporsi_bahasa(transkripsi)
    with open(lokasi_suara, "rb") as berkas_suara:
        audio_b64 = base64.b64encode(berkas_suara.read()).decode("utf-8")

    respons = JSONResponse(
        {
            "status": "success",
            "session_id": uuid.uuid4().hex,
            "mode": pilihan_mode,
            "user_text": transkripsi,
            "transcription": transkripsi,
            "normalized_text": teks_normalisasi,
            "language_tags": tag_label,
            "language_ratios": persentase,
            "llm_response": jawaban_llm,
            "response_text": jawaban_llm,
            "audio_base64": audio_b64,
        }
    )

    if back_task is not None:
        respons.background = back_task

    return respons


@app.get("/")
def check_server_status():
    return {"message": "Server FastAPI S2S Aktif. Gunakan POST ke /voice-chat."}


@app.post("/voice-chat")
@app.post("/app")
async def proses_voice_chat(
    file: UploadFile = File(...),
    mode: str = "preserve",
    format: str = "file",
):
    ekstensi = get_file_ext(file.filename)
    berkas_unggah = os.path.join(TEMP_DIR, f"inp_{uuid.uuid4()}{ekstensi}")
    berkas_sintesis = None

    try:
        isi_berkas = await file.read()
        if not isi_berkas:
            raise HTTPException(status_code=400, detail="Data audio kosong.")

        with open(berkas_unggah, "wb") as f_temp:
            f_temp.write(isi_berkas)

        # 1. Jalankan Speech-to-Text
        hasil_transkripsi = transcribe_audio_file(berkas_unggah)
        print(f"[FASTAPI-STT] Hasil: '{hasil_transkripsi}'")
        if hasil_transkripsi.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=hasil_transkripsi)

        # 2. Lakukan Normalisasi Kata Baku
        teks_normal = normalize_text(hasil_transkripsi)
        prompt_final = teks_normal if mode == "normalized" else hasil_transkripsi
        
        # 3. Minta Jawaban dari Model LLM (Gemini)
        print(f"[FASTAPI-LLM] Mengirim prompt: '{prompt_final}'")
        jawaban_ai = generate_response(prompt_final)
        print(f"[FASTAPI-LLM] Jawaban: '{jawaban_ai}'")
        if jawaban_ai.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=jawaban_ai)

        # 4. Normalisasi Angka & Kata Sebelum Disintesis
        teks_sintesis_bersih = normalize_text(jawaban_ai)

        # 5. Jalankan Text-to-Speech
        berkas_sintesis = transcribe_text_to_speech(teks_sintesis_bersih)
        
        # Hapus berkas temporary di latar belakang agar hemat penyimpanan
        task_pembersihan = BackgroundTask(lambda: [safe_delete(berkas_unggah), safe_delete(berkas_sintesis)])

        if format == "json":
            return susun_respons_json(
                transkripsi=hasil_transkripsi,
                jawaban_llm=jawaban_ai,
                teks_normalisasi=teks_normal,
                pilihan_mode=mode,
                lokasi_suara=berkas_sintesis,
                back_task=task_pembersihan,
            )

        return FileResponse(
            berkas_sintesis,
            media_type="audio/wav",
            filename="response.wav",
            headers={
                "X-Transcript": hasil_transkripsi.encode("ascii", "ignore").decode(),
                "X-LLM-Response": jawaban_ai.encode("ascii", "ignore").decode(),
            },
            background=task_pembersihan,
        )
    except HTTPException:
        safe_delete(berkas_unggah)
        safe_delete(berkas_sintesis)
        raise
    except Exception as error:
        safe_delete(berkas_unggah)
        safe_delete(berkas_sintesis)
        print(f"\n[FASTAPI-ERROR] Terjadi kegagalan pipeline:")
        print(f"Detail: {error}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(error)}") from error


@app.post("/debug-text")
async def check_debug_text(file: UploadFile = File(...)):
    """Endpoint diagnosa untuk debugging cepat."""
    ekstensi = get_file_ext(file.filename)
    berkas_debug = os.path.join(TEMP_DIR, f"dbg_{uuid.uuid4()}{ekstensi}")
    try:
        with open(berkas_debug, "wb") as f_temp:
            f_temp.write(await file.read())
        transkripsi = transcribe_audio_file(berkas_debug)
        jawaban = generate_response(transkripsi) if not transkripsi.startswith("[ERROR]") else ""
        return JSONResponse({"transcript": transkripsi, "response": jawaban})
    finally:
        safe_delete(berkas_debug)