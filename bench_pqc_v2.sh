#!/bin/bash
CONCURRENCY=$1        
TIME_SEC=5            
HOST="localhost:4433"

if [ -z "$1" ]; then
    echo "Usage: ./bench_pqc_v2.sh <number_of_processes>"
    exit 1
fi

export PATH=$HOME/openssl-3.5-cupqc/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openssl-3.5-cupqc/lib64:$LD_LIBRARY_PATH

echo "---------------------------------------------------"
echo "Benchmarking with $CONCURRENCY processes..."
echo "Target: $HOST (Auto-negotiating PQC)"
echo "---------------------------------------------------"

rm -f results.txt

for ((i=1; i<=CONCURRENCY; i++)); do
    (
        # REMOVED -groups flag because s_time doesn't support it.
        # It will use the default hybrid group (X25519MLKEM768).
        output=$(openssl s_time -connect $HOST -new -time $TIME_SEC -www 2>&1)
        
        conn=$(echo "$output" | grep "connections/user sec" | awk '{print $4}')
        
        if [ -z "$conn" ]; then
            echo "ERROR" >> results.txt
        else
            echo $conn >> results.txt
        fi
    ) &
done

wait

total_ops=0
errors=0
while read p; do
  if [ "$p" == "ERROR" ]; then
      errors=$((errors+1))
  else
      total_ops=$(echo "$total_ops + $p" | bc)
  fi
done < results.txt

echo "FINAL RESULT:"
echo "Total Throughput: $total_ops handshakes/sec"
echo "Failed Processes: $errors"
