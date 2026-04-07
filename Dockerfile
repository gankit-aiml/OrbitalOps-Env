# Use the official Meta-provided base image from your boilerplate
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set the working directory to /app (Simple and flat)
WORKDIR /app

# Ensure we are using the root user to avoid permission issues in the validator
USER root

# Install system dependencies (git is often needed for openenv-core)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# We use the pip installed in the base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into /app
COPY . .

# CRITICAL: Set PYTHONPATH so 'server' and 'models' are discoverable
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Hugging Face mandatory port
EXPOSE 7860

# Run the server on the MANDATORY port 7860
# We point directly to server.app:app
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]