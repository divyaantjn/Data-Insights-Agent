FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System deps for Kaleido/Chromium (Debian/Ubuntu names)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libxkbcommon0 \
    libxcb1 \
    libxi6 \
    libxss1 \
    libxshmfence1 \
    libxcursor1 \
    libx11-xcb1 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libdrm2 \
    fonts-dejavu-core \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-freeze.txt ./
RUN pip install --no-cache-dir -r requirements-freeze.txt

# Install Playwright browsers as root, then fix ownership
RUN python -m playwright install chromium firefox webkit && \
    mkdir -p /home/appuser && \
    cp -r /root/.cache /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser

# Download Chromium used by kaleido
RUN python - <<'PY'
import kaleido
print('Fetching Chromium via kaleido...')
kaleido.get_chrome_sync()
print('Done')
PY

COPY . .

# Give appuser ownership of /app only
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Playwright browser path for non-root user
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
