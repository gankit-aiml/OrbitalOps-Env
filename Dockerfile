# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install uv
RUN pip install uv --no-cache-dir

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no project itself yet, just deps)
RUN uv sync --frozen --no-install-project --no-cache

# Copy the rest of the project
COPY . .

# Install the project itself
RUN uv sync --frozen --no-cache

EXPOSE 7860

# Run the server via the [project.scripts] entry point
CMD ["uv", "run", "server", "--host", "0.0.0.0", "--port", "7860"]
