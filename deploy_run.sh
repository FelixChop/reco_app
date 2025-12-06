#!/bin/bash

# Navigate to backend directory
cd backend

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the server
# We use 0.0.0.0 to be accessible externally
# Port 8000 is standard
echo "Starting server on port 8000..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000
