# Base image
FROM openjdk:11-jdk-slim

# Set environment variables for Spark, Python, and Java
ENV SPARK_VERSION=3.3.2 \
    HADOOP_VERSION=3 \
    JAVA_HOME=/usr/local/openjdk-11 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3 \
    PATH=$JAVA_HOME/bin:$PATH

# Install Python, tools, build dependencies, and other utilities
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    wget \
    curl \
    vim \
    software-properties-common \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    libomp-dev \
    procps \
    python3-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set Spark environment variables
ENV SPARK_HOME=/opt/spark \
    PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# Set build argument to determine the service
ARG SERVICE
ENV SERVICE=${SERVICE}

# Copy and install the appropriate requirements file
COPY requirements/requirements_${SERVICE}.txt /tmp/requirements.txt
RUN pip install numpy scipy Cython && pip install --no-cache-dir -r /tmp/requirements.txt

# Create the working directory
WORKDIR /app

# Directly copy the files for each service
ARG SERVICE
COPY src/configs/config.yaml /app/src/configs/

# For ALS
COPY src/algorithms/als /app/src/algorithms/als

# For CBRS
COPY src/algorithms/cbrs /app/src/algorithms/cbrs

# For Fetching
COPY src/data_management/fetch_mind.py /app/src/data_management/fetch_mind.py


# Default command for each service
CMD bash -c "\
    if [ '${SERVICE}' = 'als' ]; then \
        python3 -m src.algorithms.als.run_train_als; \
    elif [ '${SERVICE}' = 'cbrs' ]; then \
        jupyter notebook --no-browser --allow-root; \
    elif [ '${SERVICE}' = 'fetching' ]; then \
        python3 -m src.data_management.fetch_mind; \
    else \
        echo 'Invalid SERVICE specified: ${SERVICE}' && exit 1; \
    fi"
