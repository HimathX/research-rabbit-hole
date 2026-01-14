# Lightweight Streamlit frontend application
# Uses only client-side dependencies for minimal image size
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /sbin/nologin appuser

# Install Python dependencies (lightweight client-only packages)
RUN pip install --no-cache-dir \
    streamlit==1.40.2 \
    pydantic>=2.10.1 \
    python-dotenv>=1.0.0 \
    httpx>=0.28.0

# Copy application source code
COPY src/streamlit_app.py ./src/
COPY src/client/ ./src/client/
COPY src/schema/ ./src/schema/

# Set proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Configure Streamlit to run in server mode
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false

# Health check - verify Streamlit is responding
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--client.showErrorDetails=false"]
