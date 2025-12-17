#include "cupqc_shim.hpp"
#include <cupqc/pk.hpp>
#include <cuda_runtime.h>
#include <vector>

using namespace cupqc;

// --- DESCRIPTORS ---
// We use Block() as it is the only supported execution mode
using Keygen768 = decltype(ML_KEM_768{} + Function<function::Keygen>() + Block() + BlockDim<256>());
using Encaps768 = decltype(ML_KEM_768{} + Function<function::Encaps>() + Block() + BlockDim<256>());
using Decaps768 = decltype(ML_KEM_768{} + Function<function::Decaps>() + Block() + BlockDim<256>());

// Helper macro
#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) return 0; }

// --- KERNELS ---

__global__ void kernel_keygen(uint8_t* pk, uint8_t* sk, uint8_t* entropy, uint8_t* workspace) {
    // FIX: Allocate shared memory (Required by API)
    __shared__ uint8_t smem[Keygen768::shared_memory_size];
    
    // FIX: Pass smem as the last argument
    Keygen768().execute(pk, sk, entropy, workspace, smem);
}

__global__ void kernel_encaps(uint8_t* ct, uint8_t* ss, const uint8_t* pk, uint8_t* entropy, uint8_t* workspace) {
    __shared__ uint8_t smem[Encaps768::shared_memory_size];
    // Note: Argument order for Encaps is (ct, ss, pk, entropy, workspace, smem)
    Encaps768().execute(ct, ss, pk, entropy, workspace, smem);
}

__global__ void kernel_decaps(uint8_t* ss, const uint8_t* ct, const uint8_t* sk, uint8_t* workspace) {
    __shared__ uint8_t smem[Decaps768::shared_memory_size];
    // Note: Argument order for Decaps is (ss, ct, sk, workspace, smem)
    Decaps768().execute(ss, ct, sk, workspace, smem);
}

// --- HOST FUNCTIONS ---

extern "C" int cupqc_shim_keygen_768(uint8_t *pk, uint8_t *sk) {
    uint8_t *d_pk, *d_sk, *d_entropy, *d_workspace;
    
    // Allocate host entropy
    std::vector<uint8_t> h_entropy(Keygen768::entropy_size);
    for(auto &b : h_entropy) b = rand() % 255;

    // Allocate Device Memory
    CUDA_CHECK(cudaMalloc(&d_pk, Keygen768::public_key_size));
    CUDA_CHECK(cudaMalloc(&d_sk, Keygen768::secret_key_size));
    CUDA_CHECK(cudaMalloc(&d_entropy, Keygen768::entropy_size));
    CUDA_CHECK(cudaMalloc(&d_workspace, Keygen768::workspace_size));

    // Copy entropy
    CUDA_CHECK(cudaMemcpy(d_entropy, h_entropy.data(), Keygen768::entropy_size, cudaMemcpyHostToDevice));

    // Launch Kernel (1 Batch)
    kernel_keygen<<<1, 256>>>(d_pk, d_sk, d_entropy, d_workspace);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy Back
    CUDA_CHECK(cudaMemcpy(pk, d_pk, Keygen768::public_key_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sk, d_sk, Keygen768::secret_key_size, cudaMemcpyDeviceToHost)); 

    // Cleanup
    cudaFree(d_pk); cudaFree(d_sk); cudaFree(d_entropy); cudaFree(d_workspace);
    return 1;
}

extern "C" int cupqc_shim_encaps_768(uint8_t *ct, uint8_t *ss, const uint8_t *pk) {
    uint8_t *d_ct, *d_ss, *d_pk, *d_entropy, *d_workspace;

    std::vector<uint8_t> h_entropy(Encaps768::entropy_size);
    for(auto &b : h_entropy) b = rand() % 255;

    CUDA_CHECK(cudaMalloc(&d_ct, Encaps768::ciphertext_size));
    CUDA_CHECK(cudaMalloc(&d_ss, Encaps768::shared_secret_size));
    CUDA_CHECK(cudaMalloc(&d_pk, Encaps768::public_key_size));
    CUDA_CHECK(cudaMalloc(&d_entropy, Encaps768::entropy_size));
    CUDA_CHECK(cudaMalloc(&d_workspace, Encaps768::workspace_size));

    CUDA_CHECK(cudaMemcpy(d_pk, pk, Encaps768::public_key_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_entropy, h_entropy.data(), Encaps768::entropy_size, cudaMemcpyHostToDevice));

    kernel_encaps<<<1, 256>>>(d_ct, d_ss, d_pk, d_entropy, d_workspace);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ct, d_ct, Encaps768::ciphertext_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ss, d_ss, Encaps768::shared_secret_size, cudaMemcpyDeviceToHost));

    cudaFree(d_ct); cudaFree(d_ss); cudaFree(d_pk); cudaFree(d_entropy); cudaFree(d_workspace);
    return 1;
}

extern "C" int cupqc_shim_decaps_768(uint8_t *ss, const uint8_t *ct, const uint8_t *sk) {
    uint8_t *d_ss, *d_ct, *d_sk, *d_workspace;

    CUDA_CHECK(cudaMalloc(&d_ss, Decaps768::shared_secret_size));
    CUDA_CHECK(cudaMalloc(&d_ct, Decaps768::ciphertext_size));
    CUDA_CHECK(cudaMalloc(&d_sk, Decaps768::secret_key_size));
    CUDA_CHECK(cudaMalloc(&d_workspace, Decaps768::workspace_size));

    CUDA_CHECK(cudaMemcpy(d_ct, ct, Decaps768::ciphertext_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sk, sk, Decaps768::secret_key_size, cudaMemcpyHostToDevice)); 

    kernel_decaps<<<1, 256>>>(d_ss, d_ct, d_sk, d_workspace);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ss, d_ss, Decaps768::shared_secret_size, cudaMemcpyDeviceToHost));

    cudaFree(d_ss); cudaFree(d_ct); cudaFree(d_sk); cudaFree(d_workspace);
    return 1;
}
