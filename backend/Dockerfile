# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /backend

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    libgraphviz-dev \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install claat
RUN curl -LO https://github.com/googlecodelabs/tools/releases/latest/download/claat-linux-amd64 \
    && mv claat-linux-amd64 /usr/local/bin/claat \
    && chmod +x /usr/local/bin/claat

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy the application code
COPY . .

# Add /app to PYTHONPATH
ENV PYTHONPATH=/app/

# Expose the port that the app runs on
EXPOSE 8000

WORKDIR /
# Set the entrypoint command
CMD ["sh", "-c", "echo 'Current working directory 1:' $(pwd),$(ls -a) && python -m backend.run_service"]
#CMD ["python", "-m", "backend.run_service"]

