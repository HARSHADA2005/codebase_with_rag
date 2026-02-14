#!/bin/bash

# RAG Application Quick Start Script
# This script activates the virtual environment and runs the Streamlit app

echo "🚀 Starting RAG Assistant..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if API key is configured
if ! grep -q "OPENAI_API_KEY.*sk-" .streamlit/secrets.toml 2>/dev/null; then
    echo "⚠️  Warning: OpenAI API key not configured!"
    echo "Please add your API key to .streamlit/secrets.toml"
    echo ""
fi

# Run the application
echo "✅ Starting Streamlit application..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo ""
streamlit run app.py
