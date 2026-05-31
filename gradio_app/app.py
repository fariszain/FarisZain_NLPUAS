import base64
import os
import tempfile
import uuid
import requests
import scipy.io.wavfile
import gradio as gr

# URL backend FastAPI
URL_BACKEND = os.getenv("FASTAPI_URL", "http://localhost:8000/voice-chat")


def bangkitkan_status_html(status="idle", deskripsi="Siap menerima input audio..."):
    """Membuat widget status dengan warna indikator dinamis."""
    css_status = {
        "idle": "idle-state",
        "processing": "running-state",
        "success": "success-state",
        "error": "error-state"
    }.get(status, "idle-state")
    
    return f"""
    <div class="panel-status {css_status}">
        <span class="dot-status"></span>
        <span class="teks-status">{deskripsi}</span>
    </div>
    """


def format_lang_chips(kamus_tag):
    """Menampilkan tag bahasa sebagai chip linguistik yang cantik."""
    if not kamus_tag:
        return "<div class='empty-data-note'>Belum ada data bahasa.</div>"
    if isinstance(kamus_tag, str):
        return f"<div class='tag-cloud-custom'>{kamus_tag}</div>"
        
    daftar_tag = []
    items = kamus_tag.items() if isinstance(kamus_tag, dict) else enumerate(kamus_tag)
    for bhs, proporsi in items:
        daftar_tag.append(f"<span class='chip-bhs'><b>{bhs}</b>: {proporsi}</span>")
    return f"<div class='tag-cloud-custom'>{''.join(daftar_tag)}</div>"


def format_ratio_text(rasio_bahasa):
    """Memformat rasio distribusi bahasa menjadi teks informatif."""
    if not rasio_bahasa:
        return ""
    if isinstance(rasio_bahasa, str):
        return rasio_bahasa
    peta_nama = {"IND": "Indonesia", "ID": "Indonesia", "EN": "Inggris", "AR": "Arab", "ID-Slang": "Slang"}
    return "  |  ".join(f"{peta_nama.get(bhs, bhs)}: {proporsi}" for bhs, proporsi in rasio_bahasa.items())


def tulis_wav_temporer(rate_audio, data_audio):
    """Menyimpan array numpy audio ke file WAV sementara."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f_tmp:
        scipy.io.wavfile.write(f_tmp.name, rate_audio, data_audio)
        return f_tmp.name


def simpan_audio_b64(b64_data, id_sesi):
    """Mendekode string base64 kembali menjadi file audio WAV."""
    path_hasil = os.path.join(tempfile.gettempdir(), f"output_voice_{id_sesi}.wav")
    with open(path_hasil, "wb") as f_out:
        f_out.write(base64.b64decode(b64_data))
    return path_hasil


def alur_pemrosesan_s2s(rekaman_audio, mode_respons):
    """
    Mengontrol jalannya orkestrasi pipeline S2S dengan memanggil backend API.
    """
    if rekaman_audio is None:
        return (
            None, "", "",
            format_lang_chips(None), "", "",
            bangkitkan_status_html("idle", "Gagal: Harap rekam atau unggah audio terlebih dahulu."),
            "Audio kosong. Masukkan suara lalu tekan tombol analisis.",
        )

    laju_sampel, array_audio = rekaman_audio
    path_unggah = tulis_wav_temporer(laju_sampel, array_audio)

    try:
        with open(path_unggah, "rb") as berkas_kirim:
            files = {"file": ("audio_input.wav", berkas_kirim, "audio/wav")}
            data = {"mode": mode_respons}
            
            respons_api = requests.post(
                f"{URL_BACKEND}?format=json", files=files, data=data, timeout=120
            )

        if respons_api.status_code != 200:
            return (
                None, "", "",
                format_lang_chips("<span class='text-danger'>Kesalahan Backend API.</span>"),
                "", "",
                bangkitkan_status_html("error", f"Gagal: Backend error HTTP {respons_api.status_code}."),
                respons_api.text,
            )

        data_json = respons_api.json()
        if data_json.get("status") == "error":
            return (
                None, "", "",
                format_lang_chips("<span class='text-danger'>Gagal memproses file.</span>"),
                "", "",
                bangkitkan_status_html("error", data_json.get("message", "Terjadi kesalahan.")),
                data_json.get("message", "Error tidak diketahui."),
            )

        id_sesi = data_json.get("session_id", uuid.uuid4().hex)
        path_output_wav = None
        if data_json.get("audio_base64"):
            path_output_wav = simpan_audio_b64(data_json["audio_base64"], id_sesi)

        transkripsi = data_json.get("transcription") or ""
        normalisasi = data_json.get("normalized_text") or ""
        tags_html = format_lang_chips(data_json.get("language_tags"))
        ratios_teks = format_ratio_text(data_json.get("language_ratios"))
        respons_llm = data_json.get("llm_response") or ""

        return (
            path_output_wav,
            transkripsi,
            normalisasi,
            tags_html,
            ratios_teks,
            respons_llm,
            bangkitkan_status_html("success", "Respons sukses dibuat!"),
            f"Analisis selesai untuk sesi {id_sesi[:8]}...",
        )

    except requests.exceptions.Timeout:
        return (
            None, "", "",
            format_lang_chips("<span class='text-danger'>Batas waktu habis.</span>"),
            "", "",
            bangkitkan_status_html("error", "Batas waktu request ke backend habis."),
            "Audio terlalu panjang atau server kehabisan memori.",
        )
    except requests.exceptions.ConnectionError:
        return (
            None, "", "",
            format_lang_chips("<span class='text-danger'>Koneksi terputus.</span>"),
            "", "",
            bangkitkan_status_html("error", "Tidak dapat menghubungi server backend FastAPI."),
            "Pastikan backend berjalan di http://127.0.0.1:8000.",
        )
    except Exception as err:
        return (
            None, "", "",
            format_lang_chips("<span class='text-danger'>Error internal.</span>"),
            "", "",
            bangkitkan_status_html("error", "Terjadi kegagalan pemrosesan."),
            str(err),
        )
    finally:
        if os.path.exists(path_unggah):
            os.remove(path_unggah)


# ---------------------------------------------------------------------------
# PREMIUM GLASSMORPHISM THEME
# ---------------------------------------------------------------------------
tema_kustom = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Space Grotesk"), "Inter", "sans-serif"],
).set(
    body_background_fill="#0b0f19",
    body_background_fill_dark="#0b0f19",
    body_text_color="#f4f4f5",
    body_text_color_subdued="#a1a1aa",
    
    input_background_fill="#111827",
    input_border_color="#1f2937",
    input_border_color_focus="#10b981",
    
    block_background_fill="rgba(17, 24, 39, 0.7)",
    block_border_color="rgba(255, 255, 255, 0.05)",
    block_shadow="0 25px 50px -12px rgba(0, 0, 0, 0.5)",
    block_radius="24px",
    
    button_primary_background_fill="linear-gradient(135deg, #10b981, #059669)",
    button_primary_text_color="#ffffff",
    
    button_secondary_background_fill="rgba(255, 255, 255, 0.05)",
    button_secondary_text_color="#f4f4f5",
)

css_styling = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

body {
  background: radial-gradient(circle at 0% 0%, #111827 0%, #0b0f19 50%, #030712 100%) !important;
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Outer layout tweaks */
.gradio-container {
  max-width: 1250px !important;
  margin: 40px auto !important;
  padding: 0 20px !important;
}

/* Glassmorphism containers */
.glass-card {
  background: rgba(17, 24, 39, 0.5) !important;
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.05) !important;
  border-radius: 24px !important;
  padding: 24px !important;
  box-shadow: 0 20px 40px rgba(0,0,0,0.3) !important;
}

/* Beautiful custom Siri orb */
.orb-container {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 30px 0;
  position: relative;
}

.pulsing-ai-orb {
  width: 90px;
  height: 90px;
  border-radius: 50%;
  background: radial-gradient(circle, #10b981 0%, #06b6d4 70%, transparent 100%);
  filter: blur(2px);
  box-shadow: 0 0 35px rgba(16, 185, 129, 0.6);
  animation: orb-pulse 2.5s infinite ease-in-out;
  position: relative;
}

.pulsing-ai-orb::after {
  content: '';
  position: absolute;
  top: -10px; left: -10px; right: -10px; bottom: -10px;
  border-radius: 50%;
  border: 2px solid rgba(6, 182, 212, 0.3);
  animation: orb-ring-expand 2.5s infinite ease-out;
}

@keyframes orb-pulse {
  0% { transform: scale(0.92); opacity: 0.8; box-shadow: 0 0 25px rgba(16, 185, 129, 0.5); }
  50% { transform: scale(1.08); opacity: 1; box-shadow: 0 0 45px rgba(6, 182, 212, 0.8); }
  100% { transform: scale(0.92); opacity: 0.8; box-shadow: 0 0 25px rgba(16, 185, 129, 0.5); }
}

@keyframes orb-ring-expand {
  0% { transform: scale(0.85); opacity: 1; }
  100% { transform: scale(1.4); opacity: 0; }
}

/* Status design */
.panel-status {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  border-radius: 14px;
  font-size: 13px;
  font-weight: 600;
  border: 1px solid rgba(255,255,255,0.05);
  margin-bottom: 20px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.panel-status.idle-state { background: rgba(31, 41, 55, 0.5); color: #9ca3af; }
.panel-status.running-state { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
.panel-status.success-state { background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
.panel-status.error-state { background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }

.dot-status {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: currentColor;
  display: inline-block;
}

/* Custom chips for languages */
.tag-cloud-custom {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 5px;
}

.chip-bhs {
  font-size: 11px;
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.3);
  color: #34d399;
  padding: 6px 12px;
  border-radius: 99px;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.empty-data-note {
  font-size: 12px;
  color: #6b7280;
  font-style: italic;
}

/* Branding details */
.branding-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 26px;
  font-weight: 700;
  background: linear-gradient(135deg, #ffffff 30%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 2px;
}

.branding-subtitle {
  font-size: 11px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  font-weight: 600;
}

/* Modern chat bubbles */
.chat-bubble-container {
  display: flex;
  flex-direction: column;
  gap: 15px;
  padding: 10px 0;
}

.chat-bubble {
  border-radius: 18px;
  padding: 16px 20px;
  max-width: 85%;
  font-size: 14.5px;
  line-height: 1.6;
}

.chat-bubble.assistant {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.chat-bubble-header {
  font-size: 11px;
  color: #6b7280;
  font-weight: 700;
  text-transform: uppercase;
  margin-bottom: 4px;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.assistant-indicator-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #10b981;
  display: inline-block;
}

/* Layout headers */
.layout-header-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 17px;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  padding-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}
"""

with gr.Blocks(theme=tema_kustom, css=css_styling, title="Nexus Speech Suite") as demo:
    with gr.Column():
        
        # Upper Brand Bar
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML("""
                <div>
                    <h1 class="branding-title">NEXUS VOICE</h1>
                    <span class="branding-subtitle">Multilingual S2S Intelligence Suite</span>
                </div>
                """)
            with gr.Column(scale=4, elem_id="status-box-parent"):
                status_box = gr.HTML(value=bangkitkan_status_html())

        # Main Workspace - Two-Column Split
        with gr.Row(equal_height=True):
            
            # LEFT SIDEBAR: DIAGNOSTICS & TELEMETRY
            with gr.Column(scale=4, elem_classes="glass-card"):
                gr.HTML("""
                <div class="layout-header-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    Diagnostics & LID
                </div>
                """)
                
                user_teks_mentah = gr.Textbox(
                    label="Speech-to-Text Transcription",
                    interactive=False,
                    placeholder="Ujaran terdeteksi akan muncul di sini.",
                )
                
                user_teks_normal = gr.Textbox(
                    label="Normalized Lexicon",
                    interactive=False,
                    placeholder="Teks normalisasi angka...",
                )
                
                gr.HTML("<span style='font-size: 12px; color: #a1a1aa; font-weight: 600;'>Code-Switching Detection</span>")
                label_tag_bhs = gr.HTML(value=format_lang_chips(None))
                
                rasio_bhs_teks = gr.Textbox(
                    label="Language Ratio",
                    interactive=False,
                    placeholder="Rasio kontribusi bahasa...",
                )
                
                status_log_detail = gr.Textbox(
                    label="System Trace Logs",
                    interactive=False,
                    value="Nexus Ready.",
                )

            # RIGHT PANEL: THE INTERACTIVE CONSOLE (SIRI-LIKE CHAT HUB)
            with gr.Column(scale=8, elem_classes="glass-card"):
                gr.HTML("""
                <div class="layout-header-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    Voice Chat Companion
                </div>
                """)
                
                # Dynamic Siri-like Glowing Orb
                gr.HTML("""
                <div class="orb-container">
                    <div class="pulsing-ai-orb"></div>
                </div>
                """)
                
                audio_masukan = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Ucapkan pertanyaan Anda di bawah ini:",
                )
                
                opsi_mode = gr.Radio(
                    choices=[("Preserve Mode (Bahasa Campuran)", "preserve"), ("Normalized Mode (Baku)", "normalized")],
                    value="preserve",
                    label="Format Respons Teks",
                )
                
                proses_btn = gr.Button("KIRIM SUARA", variant="primary")
                
                # Chat Output Panel
                with gr.Column(elem_classes="chat-bubble-container"):
                    gr.HTML("""
                    <div class="chat-bubble assistant">
                        <div class="chat-bubble-header">
                            <span class="assistant-indicator-dot"></span>
                            NEXUS ASSISTANT
                        </div>
                        Tanyakan apa saja seputar panduan umrah, rute penerbangan ke Jeddah, kunjungan ke Madinah, atau visa perjalanan Arab Saudi.
                    </div>
                    """)
                    
                    respons_teks_llm = gr.Textbox(
                        label="Respons Teks Asisten",
                        interactive=False,
                        placeholder="Respons teks akan muncul di sini.",
                        lines=3,
                    )
                    
                    audio_balasan = gr.Audio(
                        label="Respons Suara Asisten (Natural Audio)",
                        interactive=False,
                    )

        # Footer
        gr.HTML("""
        <div style="text-align: center; font-size: 11px; color: #4b5563; margin-top: 30px; letter-spacing: 1px;">
            SPEECH-TO-SPEECH NLP PIPELINE • POWERED BY GEMINI & MICROSOFT NEURAL TTS
        </div>
        """)

    # Event Mapping
    proses_btn.click(
        fn=alur_pemrosesan_s2s,
        inputs=[audio_masukan, opsi_mode],
        outputs=[
            audio_balasan,
            user_teks_mentah,
            user_teks_normal,
            label_tag_bhs,
            rasio_bhs_teks,
            respons_teks_llm,
            status_box,
            status_log_detail,
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)