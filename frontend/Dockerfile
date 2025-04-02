# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    libgraphviz-dev \
    curl \
    libpq-dev \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Go for ARM64 and set up PATH
RUN curl -LO https://golang.org/dl/go1.21.1.linux-arm64.tar.gz \
    && tar -C /usr/local -xzf go1.21.1.linux-arm64.tar.gz \
    && rm go1.21.1.linux-arm64.tar.gz \
    && export PATH=$PATH:/usr/local/go/bin

ENV PATH="/usr/local/go/bin:$PATH"

# Build claat from source
RUN git clone https://github.com/googlecodelabs/tools.git \
    && cd tools/claat \
    && go build -o claat \
    && mv claat /usr/local/bin/ \
    && chmod +x /usr/local/bin/claat \
    && rm -rf /tools

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy the application code
COPY . .

# Expose the port that Streamlit and claat run on
EXPOSE 8501
EXPOSE 9090

# Set the entrypoint command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
