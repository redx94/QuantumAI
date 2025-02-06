#!/bin/bash

# Script to build and deploy QuantumAI using Docker

## Step 1: Clean old containers without affecting any active instances
docker compose -a quantumai || decl
## Step 2: Build the docker with the latest code and required dependencies
docker build -f dockerfile -t quantumai:latest .
## Step 3: Run the container and map to the host machine
docker run -d quantumai -p 8000:8000 -t quantumai:latest
