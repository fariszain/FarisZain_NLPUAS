#!/bin/bash

# ==============================================================================
# SCRIPT UTILITY: MENJALANKAN BACKEND & FRONTEND UAS S2S CHATBOT
# ==============================================================================

PROJECT_DIR="/home/zeyn/Documents/UAS_PrakNLP_Speech2Text"
PYTHON_BIN="/home/zeyn/Documents/all/bin/python"

echo "=================================================================="
echo "          NEXUS S2S ASSISTANT: MANAGEMENT SERVER SCRIPT           "
echo "=================================================================="

# 1. Bersihkan port jika masih dipakai proses sebelumnya
echo "[1/3] Memeriksa dan mematikan sisa proses lama di port 8000 & 7860..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 7860/tcp 2>/dev/null
sleep 2

# 2. Jalankan FastAPI Backend
echo "[2/3] Memulai server API FastAPI (port 8000)..."
cd "$PROJECT_DIR"
$PYTHON_BIN -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "      FastAPI berjalan di latar belakang (PID: $FASTAPI_PID, log: fastapi.log)"
sleep 3

# 3. Jalankan Gradio App
echo "[3/3] Memulai UI dashboard Gradio (port 7860)..."
# Tunggu sebentar lalu jalankan Gradio
$PYTHON_BIN gradio_app/app.py
