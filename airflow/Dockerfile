# Use the official Airflow image as the base image
FROM apache/airflow:2.10.3

# Set up environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (e.g., build-essential, wget, curl)
USER root
RUN apt-get update && apt-get install -y \
    wget curl unzip build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Export the AIRFLOW_UID environment variable
ENV AIRFLOW_UID=50000

# Copy requirements.txt to the image and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy application code to the Airflow home directory
WORKDIR /opt/airflow
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
COPY data /opt/airflow/data
