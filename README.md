# NVIDIA cuPQC Integration for OpenSSL Core


**A high-performance, middleware-free integration of NVIDIA cuPQC directly into the OpenSSL 3.5 cryptographic core.**

---

##  Overview

This project addresses the computational bottleneck of Post-Quantum Cryptography (PQC) by offloading **ML-KEM-768** key generation and encapsulation directly to NVIDIA GPUs. 

Unlike traditional "Provider" approaches (e.g., OQS-Provider) that introduce latency via dispatch layers and intermediate libraries (`liboqs`), this solution patches the OpenSSL core (`ml_kem.c`) to communicate directly with a custom CUDA shim. This architecture minimizes stack depth and memory copying, enabling efficient hardware acceleration for TLS 1.3 handshakes.

## Key Achievement:

---
### Key Performance Results

**Benchmark Context:**
* **Algorithm:** ML-KEM-768 (Key Encapsulation)
* **Hardware:** AWS G4dn.xlarge (4 vCPUs, NVIDIA Tesla T4)
* **Optimization:** CUDA Pinned Memory (DMA) + Batching (N=64)

| Implementation | Configuration | Throughput (Ops/Sec) | Speedup vs CPU |
| :--- | :--- | :--- | :--- |
| **CPU (Baseline)** | 1 Core (Serial) | 19,031 | 1.0x |
| **CPU (Multi-Core)** | 4 Cores (100% Load) | 80,892 | 4.2x |
| **GPU (Previous)** | Standard Pageable Mem | 32,320 | 1.7x |
| **This Project** | **Pinned Memory (DMA)** | **125,963** | **6.6x** |

> **Impact:** The GPU implementation provides a **1.55x speedup** over the fully saturated 4-core Host CPU while keeping host CPU utilization below 5%.

## System Architecture

This project modifies the OpenSSL `ml_kem.c` core to interact with a custom C++ CUDA Shim. The architecture focuses on eliminating the "Transfer Penalty" over the PCIe bus.

### Data Flow
1.  **Request Aggregation:** OpenSSL requests are captured by the Shim.
2.  **Zero-Copy Staging:** Data is written directly into **Pinned Memory (Page-Locked)** using `cudaHostAlloc`, bypassing OS paging buffers.
3.  **DMA Transfer:** The GPU pulls data directly from host RAM via Direct Memory Access (DMA).
4.  **Parallel Execution:** `cuPQC` kernels process batches (64-512 ops) asynchronously.

**Comparison:**
* **Standard OQS Flow:** `TLS` -> `EVP` -> `Provider` -> `liboqs` -> `Cupqc` ->`NVIDIA GPU`
* **cuPQC Flow:** `TLS` -> `EVP` -> `Core Shim` -> `DMA (PCIe)` -> `NVIDIA GPU`

---

## Repository Structure

```text
├── benchmark_pqc           # High-concurrency throughput benchmarking tool
├── cupqc_shim.cu           # CUDA Shim: Handles Pinned Memory, DMA, and Batching
├── cupqc_shim.h            # Header interface for OpenSSL
├── openssl_patches/        # Modified OpenSSL source tree
│   └── crypto/
│       └── ml_kem/
│           └── ml_kem.c    # Core PQC logic patched for GPU offload
├── libcupqc.so             # Compiled shared object for the Shim
└── scripts/                # Build and test automation scripts
```
### Prerequisites
<ul>
<li>Hardware: NVIDIA Data Center GPU (Tesla T4, A100, etc.)</li>
<li>OS: Linux (Ubuntu 20.04/22.04 Tested)</li>
<li>Software:
<ul>
   <li>NVIDIA CUDA Toolkit 12.x</li>
   <li>NVIDIA cuPQC SDK (Early Access)</li>
   <li>OpenSSL 3.5.0 Source (Target version)</li>
</ul>
</li>
</ul>


## Build & Installation

**1. Prepare OpenSSL**

Download the clean OpenSSL 3.5.0 source to match the patch version.
```bash
wget [https://github.com/openssl/openssl/archive/refs/tags/openssl-3.5.0.tar.gz](https://github.com/openssl/openssl/archive/refs/tags/openssl-3.5.0.tar.gz)
tar xzf openssl-3.5.0.tar.gz
```
**2. Apply the Integration Patch**

Apply cupqc_integration.patch to inject the GPU offload logic into ml_kem.c.
```bash
cd openssl-3.5.0
patch -p1 < ../cupqc_integration.patch
```
**3. Build the Shim and Link**

***Note: The shim requires Device Link Time Optimization (DLTO) to link against static cuPQC libraries.***
```bash
# Example compilation flow
nvcc -c -o cupqc_shim.o cupqc_shim.cu -std=c++17 -dlto -fPIC
# Link shim with OpenSSL build configuration (refer to Makefile/Scripts)
```

### Usage & Verification

#### Enable Hardware Offload
The integration allows runtime switching between the default CPU implementation and the GPU engine using an environment variable.
```bash
# Enable GPU Offload
export ENABLE_CUPQC=1

# Disable (Fall back to Software)
unset ENABLE_CUPQC
```
**1. Compile the Shim**
Compile the CUDA Shim into a shared library with Position Independent Code (PIC).
```bash
nvcc -shared -rdc=true -dlto -o libcupqc.so cupqc_shim.cu \
  -Xcompiler -fPIC \
  -I./cupqc_sdk/include -L./cupqc_sdk/lib -lcupqc-pk -lcupqc-hash
```
### 2. Run Performance Benchmark
Use the custom benchmarking tool to validate throughput and stability.

```bash
# Set library path
export LD_LIBRARY_PATH=$PWD:$PWD/openssl-3.5.0:$LD_LIBRARY_PATH

# Run benchmark (64 concurrent jobs, aligned iterations)
./benchmark_pqc -jobs 64 -iters 200192
```
### 3. TLS 1.3 Handshake Verification
Verify functional correctness by performing a full TLS handshake using openssl s_server.

**Start Server (GPU Accelerated):**
```bash
openssl s_server -cert cert.pem -key key.pem -accept 4433 -tls1_3 -groups mlkem768
```
**Connect Client:**
```bash
openssl s_client -connect localhost:4433 -groups mlkem768
```
**Verified Output:**
```text
Negotiated TLS1.3 group: MLKEM768
Server Temp Key: ML-KEM-768
```


**Disclaimer:** This project is a research proof-of-concept. The cupqc_sdk libraries are not included in this repository and must be obtained directly from NVIDIA.

