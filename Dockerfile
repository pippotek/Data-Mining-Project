# Base image
FROM openjdk:11

# Set environment variables for Spark
ENV SPARK_VERSION=3.3.0 \
    HADOOP_VERSION=3 \
    SPARK_NLP_VERSION=5.5.1 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

# Install Python and required tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy pyspark==3.3.0
RUN pip3 install spark-nlp==5.5.1 pymongo pyyaml


ENV SPARK_VERSION=3.3.0 \
    HADOOP_VERSION=3 \
    SPARK_NLP_VERSION=5.5.1 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark \
    PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# Copy your Python application and configuration
WORKDIR /app
COPY src/TESTS /app/src/
COPY src/config.yaml /app/src/config.yaml

# Set entrypoint for running the Python application
ENTRYPOINT ["python3", "/app/src/clean_embed.py"]
