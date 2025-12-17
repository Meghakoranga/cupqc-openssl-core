# OpenSSL Core Integration with NVIDIA cuPQC

### **Direct Integration of Hardware-Accelerated Post-Quantum Cryptography (cuPQC) into OpenSSL 3.5 Core**


This project demonstrates a high-performance, middleware-free integration of **NVIDIA's cuPQC** (Post-Quantum Cryptography) SDK directly into the core of **OpenSSL 3.5.0-alpha**.

Unlike traditional approaches that use the "OQS Provider" architecture (which introduces overhead via `liboqs` and provider dispatch layers), this project modifies the OpenSSL source code (`ml_kem.c`) to communicate native C++ CUDA kernels.

**Key Achievement:**
Successfully offloaded **ML-KEM-768** key generation to an NVIDIA Tesla T4 GPU during a live TLS 1.3 handshake, achieving **~2,980 operations/second** in single-stream benchmarks.

---

##  Architecture

The architecture bypasses the standard provider mechanism to reduce stack depth and latency.

* **Standard Flow:** `TLS` -> `EVP` -> `Provider (OQS)` -> `liboqs` -> `Hardware`
* **This Project:** `TLS` -> `EVP` -> `Core (Patched)` -> `Shim` -> `NVIDIA GPU`



### Repository Structure

```text
├── benchmark_pqc          # Custom C benchmarking tool for throughput testing
├── cupqc_shim.cu          # C++ CUDA Shim bridging OpenSSL (C) and cuPQC (C++)
├── cupqc_shim.h           # Header definitions for the shim
├── cupqc_test.cu          # Standalone harness to verify GPU/Driver health
├── openssl_patches/       # Contains the modified OpenSSL source code
│   └── crypto/
│       └── ml_kem/
│           └── ml_kem.c   # The modified core file with GPU offload logic
├── cupqc_integration.patch # Diff file showing exact changes made to OpenSSL
└── scripts/               # Helper scripts (build, run, test)
```

### Prerequisites & Environment
**GPU**: NVIDIA Data Center GPU (Tesla T4, A100) or High-end Consumer GPU.
**SDKs**: NVIDIA cuPQC (Early Access), CUDA Toolkit 12.x+.
**Base Library**: OpenSSL 3.5.0-alpha (Modified Source).

### Usage & Verification
**Enable the Offload Engine**
The integration is guarded by an environment variable to allow runtime switching between CPU (Software) and GPU (Hardware).
### Throughput Benchmark
Run the custom benchmarking tool to stress-test the key generation path:
```
./benchmark_pqc
```
###  TLS 1.3 Handshake Verification
Start the server with GPU offloading enabled:
```
openssl s_server -cert server.crt -key server.key -accept 4433 -www
```
Connect a client using the Hybrid Post-Quantum group:
```
openssl s_client -connect localhost:4433 -groups X25519MLKEM768
```
**Result**: The server successfully negotiates the hybrid group, generating the ML-KEM-768 key share on the GPU.
