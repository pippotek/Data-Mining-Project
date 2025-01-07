#!/bin/bash

# Required tools: yq (a command-line YAML processor), Docker, and Docker Compose

# Define the path to the config file
CONFIG_FILE="src/configs/config.yaml"

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "yq could not be found. Please install yq to proceed."
    exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Parse the config.yaml to find services to exclude
EXCLUDED_SERVICES=()

# Loop through services in config.yaml
while IFS= read -r service; do
    # Read the enabled flag for each service
    enabled=$(yq ".services.$service.enabled" "$CONFIG_FILE")

    # If the service is disabled, add it to the exclusion list
    if [ "$enabled" != "true" ]; then
        EXCLUDED_SERVICES+=("$service")
    fi
done < <(yq ".services | keys[]" "$CONFIG_FILE")

# Build the exclude list for Docker Compose
EXCLUDE_ARGS=()
for service in "${EXCLUDED_SERVICES[@]}"; do
    EXCLUDE_ARGS+=("--scale" "$service=0")
done

# Spin up Docker Compose with the exclusion arguments
echo "Starting Docker Compose, excluding the following services: ${EXCLUDED_SERVICES[*]}"
docker compose up --build -d "${EXCLUDE_ARGS[@]}"
