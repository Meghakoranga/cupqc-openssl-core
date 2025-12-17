#!/bin/bash

# Configuration
# -----------------
CONCURRENCY=$1        # Number of parallel processes (e.g., 50, 100)
TIME_SEC=10           # How long to run the test
HOST="localhost:4433"
# -----------------

if [ -z "$1" ]; then
    echo "Usage: ./bench_pqc.sh <number_of_processes>"
    exit 1
fi

# Setup Environment (Point to your Custom OpenSSL)
export PATH=$HOME/openssl-3.5-cupqc/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openssl-3.5-cupqc/lib64:$LD_LIBRARY_PATH

echo "---------------------------------------------------"
echo "Benchmarking with $CONCURRENCY parallel processes for $TIME_SEC seconds..."
echo "Target: $HOST (Group: X25519MLKEM768)"
echo "---------------------------------------------------"

# File to store results
rm -f results.txt

# Launch parallel processes
for ((i=1; i<=CONCURRENCY; i++)); do
    (
        # Run s_time
        # -new : Force new handshake (don't reuse session keys) to stress KeyGen
        # -time : Run for X seconds
        output=$(openssl s_time -connect $HOST -new -time $TIME_SEC -www / 2>&1)
        
        # Extract the "connections/user sec" number
        conn=$(echo "$output" | grep "connections/user sec" | awk '{print $4}')
        echo $conn >> results.txt
    ) &
done

# Wait for all background processes to finish
wait

# Calculate Total Throughput
total_ops=0
while read p; do
  # Sum up the connections/sec from all processes
  total_ops=$(echo $total_ops + $p | bc)
done < results.txt

echo "---------------------------------------------------"
echo "FINAL RESULT:"
echo "Total Handshakes per Second: $total_ops"
echo "---------------------------------------------------"
