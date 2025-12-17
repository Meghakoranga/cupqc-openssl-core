#!/bin/bash

# Configuration
OPENSSL_BIN="$HOME/openssl-3.5-cupqc/bin/openssl"
CERT="server.crt"
KEY="server.key"
DURATION=10
BASE_PORT=4433
CORES=4 

echo "---------------------------------------------------"
echo "Starting GPU-Accelerated PQC Benchmark (TLS 1.3 + No Tickets)"
echo "---------------------------------------------------"

# 1. Cleanup
pkill s_server
sleep 1

# 2. Start Servers
# We use -num_tickets 0 to stop server-side generation
# We allow TLS 1.3 (removed -tls1_2) so PQC works
echo "[*] Launching $CORES server processes..."
for ((i=0; i<CORES; i++)); do
    PORT=$((BASE_PORT + i))
    $OPENSSL_BIN s_server -cert $CERT -key $KEY -accept $PORT -www -quiet -num_tickets 0 > /dev/null 2>&1 &
done

sleep 2

# 3. Run Clients
echo "[*] blasting traffic for $DURATION seconds on $CORES ports..."
pids=""
for ((i=0; i<CORES; i++)); do
    PORT=$((BASE_PORT + i))
    #$OPENSSL_BIN s_time -connect localhost:$PORT -new -time $DURATION -bugs -groups X25519MLKEM768 > bench_$PORT.log 2>&1 &
$OPENSSL_BIN s_time -connect localhost:$PORT -new -time $DURATION -bugs > bench_$PORT.log 2>&1 &
    pids="$pids $!"
done

# 4. Wait
wait $pids

# 5. Cleanup
pkill s_server

# 6. Results
echo "---------------------------------------------------"
echo "RESULTS:"
TOTAL_CONNECTIONS=0

for ((i=0; i<CORES; i++)); do
    PORT=$((BASE_PORT + i))
    
    if [ ! -s bench_$PORT.log ]; then
        echo "Port $PORT: Log empty (Benchmark failed)"
        continue
    fi

    # Parse log for successful connection count
    CONN=$(grep "connections in" bench_$PORT.log | awk '{print $1}')
    RATE=$(grep "connections/user sec" bench_$PORT.log | awk '{print $4}')
    
    if [ -z "$CONN" ]; then 
        # If grep failed, check if there was a read error printed
        ERR=$(head -n 1 bench_$PORT.log)
        echo "Port $PORT: Error -> $ERR"
        CONN=0
    else
        echo "Port $PORT: $CONN total connections ($RATE conn/sec)"
    fi
    
    TOTAL_CONNECTIONS=$((TOTAL_CONNECTIONS + CONN))
done

if [ "$DURATION" -gt 0 ]; then
    TOTAL_RATE=$(echo "scale=2; $TOTAL_CONNECTIONS / $DURATION" | bc)
else
    TOTAL_RATE="0"
fi

echo "---------------------------------------------------"
echo "TOTAL THROUGHPUT: ~ $TOTAL_RATE Handshakes/Second"
echo "---------------------------------------------------"
rm bench_*.log
