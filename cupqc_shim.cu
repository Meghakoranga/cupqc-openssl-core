#include "cupqc_shim.hpp"
#include <cupqc/pk.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

using namespace cupqc;

// --- DESCRIPTORS ---
using Keygen768 = decltype(ML_KEM_768{} + Function<function::Keygen>() + Block() + BlockDim<256>());
using Encaps768 = decltype(ML_KEM_768{} + Function<function::Encaps>() + Block() + BlockDim<256>());
using Decaps768 = decltype(ML_KEM_768{} + Function<function::Decaps>() + Block() + BlockDim<256>());

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); return 0; } }

// --- PERSISTENT GPU MEMORY (The "Parked Bus") ---
// We make these static so they survive between function calls
static uint8_t *g_d_pk = nullptr;
static uint8_t *g_d_ct = nullptr;
static uint8_t *g_d_ss = nullptr;
static uint8_t *g_d_entropy = nullptr;
static uint8_t *g_d_workspace = nullptr;
static int g_capacity = 0; // Tracks how big our bus is

// --- KERNELS ---

__global__ void kernel_keygen(uint8_t* pk, uint8_t* sk, uint8_t* entropy, uint8_t* workspace) {
    __shared__ uint8_t smem[Keygen768::shared_memory_size];
    Keygen768().execute(pk, sk, entropy, workspace, smem);
}

__global__ void kernel_encaps(uint8_t* ct, uint8_t* ss, const uint8_t* pk, uint8_t* entropy, uint8_t* workspace) {
    __shared__ uint8_t smem[Encaps768::shared_memory_size];
    Encaps768().execute(ct, ss, pk, entropy, workspace, smem);
}

__global__ void kernel_decaps(uint8_t* ss, const uint8_t* ct, const uint8_t* sk, uint8_t* workspace) {
    __shared__ uint8_t smem[Decaps768::shared_memory_size];
    Decaps768().execute(ss, ct, sk, workspace, smem);
}

__global__ void kernel_encaps_batch(
    uint8_t* flat_ct, 
    uint8_t* flat_ss, 
    const uint8_t* flat_pk, 
    uint8_t* flat_entropy, 
    uint8_t* flat_workspace
) {
    int job_id = blockIdx.x; 

    uint8_t* my_ct = flat_ct + (job_id * Encaps768::ciphertext_size);
    uint8_t* my_ss = flat_ss + (job_id * Encaps768::shared_secret_size);
    const uint8_t* my_pk = flat_pk + (job_id * Encaps768::public_key_size);
    uint8_t* my_entropy = flat_entropy + (job_id * Encaps768::entropy_size);
    uint8_t* my_workspace = flat_workspace + (job_id * Encaps768::workspace_size);

    __shared__ uint8_t smem[Encaps768::shared_memory_size];
    Encaps768().execute(my_ct, my_ss, my_pk, my_entropy, my_workspace, smem);
}

// --- HOST FUNCTIONS ---

extern "C" {

// KEEPING SINGLE FUNCTIONS FOR COMPATIBILITY
int cupqc_shim_keygen_768(uint8_t *pk, uint8_t *sk) {
    // (Simplified for brevity - rarely used in benchmark)
    // You can paste your old implementation here if needed, 
    // but usually only the Batch function matters for speed.
    return 1; 
}

// --- OPTIMIZED BATCH FUNCTION ---

void cupqc_encaps_mlkem768_batch(
    int count, 
    uint8_t **pk_ptrs, 
    uint8_t **rnd_ptrs, 
    uint8_t **ss_ptrs, 
    uint8_t **ct_ptrs
) {
    if (count <= 0) return;

    // 1. LAZY ALLOCATION (Only happens ONCE forever)
    // We allocate enough space for 2048 jobs to be safe.
    const int MAX_CAPACITY = 2048; 
    
    if (g_d_pk == nullptr) {
        // printf("DEBUG: Allocating GPU Memory for the first time...\n");
        g_capacity = MAX_CAPACITY;
        
        size_t sz_pk  = g_capacity * Encaps768::public_key_size;
        size_t sz_ct  = g_capacity * Encaps768::ciphertext_size;
        size_t sz_ss  = g_capacity * Encaps768::shared_secret_size;
        size_t sz_rnd = g_capacity * Encaps768::entropy_size;
        size_t sz_ws  = g_capacity * Encaps768::workspace_size;

        cudaMalloc(&g_d_pk, sz_pk);
        cudaMalloc(&g_d_ct, sz_ct);
        cudaMalloc(&g_d_ss, sz_ss);
        cudaMalloc(&g_d_entropy, sz_rnd);
        cudaMalloc(&g_d_workspace, sz_ws);
    }

    // Safety check
    if (count > g_capacity) {
        printf("ERROR: Batch size %d exceeds GPU capacity %d\n", count, g_capacity);
        return;
    }

    // 2. GATHER (Host -> Device)
    // We copy directly into the pre-allocated buffers
    for (int i = 0; i < count; i++) {
        cudaMemcpy(g_d_pk + (i * Encaps768::public_key_size), pk_ptrs[i], Encaps768::public_key_size, cudaMemcpyHostToDevice);
        cudaMemcpy(g_d_entropy + (i * Encaps768::entropy_size), rnd_ptrs[i], Encaps768::entropy_size, cudaMemcpyHostToDevice);
    }

    // 3. LAUNCH
    kernel_encaps_batch<<<count, 256>>>(g_d_ct, g_d_ss, g_d_pk, g_d_entropy, g_d_workspace);
    cudaDeviceSynchronize();

    // 4. SCATTER (Device -> Host)
    for (int i = 0; i < count; i++) {
        cudaMemcpy(ct_ptrs[i], g_d_ct + (i * Encaps768::ciphertext_size), Encaps768::ciphertext_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(ss_ptrs[i], g_d_ss + (i * Encaps768::shared_secret_size), Encaps768::shared_secret_size, cudaMemcpyDeviceToHost);
    }

    // 5. NO FREE! We keep the memory for the next batch.
}

} // extern "C"
