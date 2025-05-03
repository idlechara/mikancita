# Use a pre-built Ubuntu-based devcontainer image
FROM mcr.microsoft.com/devcontainers/base:dev-ubuntu-24.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenGL library required by OpenCV
RUN apt-get update && apt-get install -y libgl1 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install GLib library required by OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Create a Python virtual environment
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Set the virtual environment as the default Python environment
ENV PATH="/app/venv/bin:$PATH"

# Create a vscode user and set permissions
RUN id -u vscode &>/dev/null || useradd -ms /bin/bash vscode && \
    mkdir -p /home/vscode && \
    chown -R vscode:vscode /home/vscode && \
    chown -R vscode:vscode /app

# Switch to the vscode user
USER vscode

# Copy the source code
COPY src/ ./src/
COPY tests/ ./tests/

# Set the entry point for the application
ENTRYPOINT ["/app/venv/bin/python", "src/main.py"]