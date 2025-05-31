#!/bin/bash

echo "🔄 Ingesting documents..."
python backend/ingest.py

echo "🚀 Starting backend..."
uvicorn backend.main:app --reload &

sleep 3

echo "🎨 Launching frontend..."
streamlit run frontend/app.py
