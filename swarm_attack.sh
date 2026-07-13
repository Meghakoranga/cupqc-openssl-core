#!/bin/bash
echo "Starting Swarm Attack (Ports 4433-4436)..."
pids=""
for port in {4433..4436}; do
    ./benchmark_tls 16 2500 $port > res_${port}.txt &
    pids="$pids $!"
done
wait $pids
echo "Attack Complete. Checking Results:"
grep "Rate" res*.txt
