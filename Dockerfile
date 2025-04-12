# Base image: Python 3.10 on slim Debian
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies required for network analysis libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libxml2-dev \
    libcairo2-dev \
    pkg-config \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for network analysis
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    networkx \
    python-louvain \
    leidenalg \
    infomap \
    cdlib \
    python-igraph \
    scikit-learn \
    tqdm

# Optional: Install additional visualization tools if needed
RUN pip install --no-cache-dir \
    plotly \
    pycairo \
    cairocffi

# Create directory for data
RUN mkdir -p /app/data
RUN mkdir -p /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: just keep container running to allow exec into it
CMD ["tail", "-f", "/dev/null"]