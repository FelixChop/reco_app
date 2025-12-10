#!/bin/bash

# Navigate to backend directory
cd backend

# Install dependencies (sur Render, elles sont déjà installées au build,
# mais on laisse cette étape pour compatibilité locale si besoin)
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the server
# On écoute sur le port imposé par Render (PORT), avec un fallback 8000 en local
PORT_TO_USE=${PORT:-8000}
echo "Starting server on port ${PORT_TO_USE}..."
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT_TO_USE}"
