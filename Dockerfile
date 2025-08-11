# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create user for security (don't run as root)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create directories for data persistence with proper permissions
RUN mkdir -p /app/data /app/logs /app/backups

# Copy requirements first for better Docker layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Add user's pip bin to PATH
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure database is available in the container
# If equipment.db exists, copy it to the data directory for persistence
RUN if [ -f /app/equipment.db ]; then \
        cp /app/equipment.db /app/data/equipment.db && \
        echo "Database copied to /app/data/equipment.db"; \
    else \
        echo "WARNING: equipment.db not found in build context"; \
        echo "The database will need to be built or mounted from a volume"; \
    fi

# Set environment variables for Docker
ENV PYTHONPATH=/app \
    LOG_DIR=/app/logs \
    DB_PATH=/app/data/equipment.db \
    BACKUPS_DIR=/app/backups \
    TESCOM_LOGO_PATH=/app/assets/tescom-logo.png \
    GRADIO_HOST=0.0.0.0 \
    GRADIO_PORT=7860

# Create volumes for data persistence
VOLUME ["/app/data", "/app/logs", "/app/backups"]

# Expose the port
EXPOSE 7860

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set the entrypoint
CMD ["python", "app.py"]
