# Use the OpenJDK 11 image as the base
FROM openjdk:11-jdk-slim

# Set environment variables for versions
ENV SPARK_VERSION=3.3.2 \
    HADOOP_VERSION=3 \
    JAVA_HOME=/usr/local/openjdk-11 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

# Update PATH with JAVA_HOME
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Python, tools, build dependencies, and procps in one layer
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set Spark environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# Copy requirements early to leverage Docker cache
COPY requirements_als.txt /tmp/requirements_als.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements_als.txt

# Install Jupyter
RUN pip install --no-cache-dir notebook

# Configure Jupyter: No token, no password (for development purposes only)
RUN jupyter notebook --generate-config \
    && echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Set the working directory
WORKDIR /app

# Copy application code
COPY . /app

# Start Jupyter by default
CMD ["jupyter", "notebook", "--no-browser", "--allow-root"]
