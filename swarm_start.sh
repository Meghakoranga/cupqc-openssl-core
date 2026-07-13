#!/bin/bash

# 1. SETUP PATHS
OPENSSL_BIN="/home/ubuntu/openssl-openssl-3.5.0/apps/openssl"
PROJECT_DIR="/home/ubuntu/cupqc_project"
export LD_LIBRARY_PATH=$PROJECT_DIR:$PROJECT_DIR/../openssl-openssl-3.5.0:$LD_LIBRARY_PATH
export ENABLE_CUPQC=1

# 2. DYNAMIC CORE DETECTION
# This detects if you have 4, 8, or 16 cores so taskset never fails.
TOTAL_CORES=$(nproc)
MAX_CORE_INDEX=$((TOTAL_CORES - 1))

echo "--- System Check ---"
echo "Detected $TOTAL_CORES CPU Cores. Using Cores 0 to $MAX_CORE_INDEX."

# 3. AGGRESSIVE CLEANUP
# This kills anything running on our ports (4433-4440)
echo "Cleaning up old server instances..."
for PORT in {4433..4440}; do
    fuser -k $PORT/tcp 2>/dev/null
done
sleep 1 # Give the OS a second to release the ports

# 4. LAUNCH SWARM
echo "Launching GPU Server Swarm..."

for i in $(seq 0 $MAX_CORE_INDEX)
do
   PORT=$((4433 + i))
   echo "Starting Server on Port $PORT (Pinned to Core $i)"
   # We use 'taskset -c' to ensure each server gets its own dedicated core
   taskset -c $i $OPENSSL_BIN s_server -accept $PORT -cert cert.pem -key key.pem -tls1_3 -groups mlkem768 -no_ticket -quiet &
done

echo "Swarm is live. Use 'nvitop' or 'nvidia-smi' to monitor."
