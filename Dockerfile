# ── Base Image ─────────────────────────────────────────────
FROM python:3.11-slim

# ── Set Working Directory ──────────────────────────────────
WORKDIR /app

# ── Install Dependencies ───────────────────────────────────
# Copy requirements first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ──────────────────────────────────
# Only code — NO data files, NO model files
COPY app.py .
COPY train.py .

# ── Expose Streamlit Port ──────────────────────────────────
EXPOSE 8501

# ── Start the App ──────────────────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]