FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir streamlit requests pandas

# Copy the app code
COPY app.py .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Healthcheck to ensure the UI is up
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]