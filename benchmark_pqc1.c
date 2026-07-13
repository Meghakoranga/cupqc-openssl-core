#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

// Define sizes for ML-KEM-768
#define CRYPTO_PUBLICKEYBYTES 1184
#define CRYPTO_CIPHERTEXTBYTES 1088
#define CRYPTO_BYTES 32
#define CRYPTO_ENTROPYBYTES 32

// Declaration of your Shim functions
extern int cupqc_shim_keygen_768(uint8_t *pk, uint8_t *sk);
extern void cupqc_encaps_mlkem768_batch(int count, uint8_t **pk_ptrs, uint8_t **rnd_ptrs, uint8_t **ss_ptrs, uint8_t **ct_ptrs);

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int jobs = 1;
    int iters = 10; // Default to small number for testing

    if (argc >= 3 && strcmp(argv[1], "-jobs") == 0) jobs = atoi(argv[2]);
    if (argc >= 5 && strcmp(argv[3], "-iters") == 0) iters = atoi(argv[4]);

    printf("Benchmarking ML-KEM-768 (SAFE MODE)\n");
    printf("Jobs: %d | Iters: %d\n", jobs, iters);

    // 1. Allocations
    printf("[HOST] Allocating Memory...\n");
    uint8_t *pk = (uint8_t *)malloc(CRYPTO_PUBLICKEYBYTES);
    uint8_t *sk = (uint8_t *)malloc(CRYPTO_PUBLICKEYBYTES); // Size doesn't matter for dummy

    // Arrays of pointers (what the Shim expects)
    uint8_t **pk_ptrs = (uint8_t **)malloc(jobs * sizeof(uint8_t *));
    uint8_t **rnd_ptrs = (uint8_t **)malloc(jobs * sizeof(uint8_t *));
    uint8_t **ss_ptrs = (uint8_t **)malloc(jobs * sizeof(uint8_t *));
    uint8_t **ct_ptrs = (uint8_t **)malloc(jobs * sizeof(uint8_t *));

    // Data buffers
    for (int i = 0; i < jobs; i++) {
        pk_ptrs[i] = pk; // Everyone uses same key for benchmark
        rnd_ptrs[i] = (uint8_t *)malloc(CRYPTO_ENTROPYBYTES);
        ss_ptrs[i] = (uint8_t *)malloc(CRYPTO_BYTES);
        ct_ptrs[i] = (uint8_t *)malloc(CRYPTO_CIPHERTEXTBYTES);
        
        // Fill entropy with dummy data (Fast!)
        for(int j=0; j<CRYPTO_ENTROPYBYTES; j++) rnd_ptrs[i][j] = (uint8_t)j;
    }

    // 2. Keygen
    printf("[HOST] Generating Keypair...\n");
    cupqc_shim_keygen_768(pk, sk);

    // 3. Loop
    printf("[HOST] Starting Loop...\n");
    double start = get_time();
    
    for (int i = 0; i < iters; i++) {
        // Print every 10th step so we know it's alive
        if (i % 10 == 0) printf("[HOST] Iteration %d/%d\n", i, iters);
        
        cupqc_encaps_mlkem768_batch(jobs, pk_ptrs, rnd_ptrs, ss_ptrs, ct_ptrs);
    }

    double end = get_time();
    printf("[HOST] Done! Total Time: %.4f sec\n", end - start);

    return 0;
}
