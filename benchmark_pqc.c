#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openssl/evp.h>
#include <openssl/err.h>

int main(int argc, char **argv) {
    // 1. Setup Algorithm
    const char *algo_name = "ML-KEM-768";
    printf("Benchmarking %s...\n", algo_name);

    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_from_name(NULL, algo_name, NULL);
    if (!ctx) {
        fprintf(stderr, "Error: Could not find algorithm '%s'.\n", algo_name);
        ERR_print_errors_fp(stderr);
        return 1;
    }

    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        fprintf(stderr, "Error: Keygen init failed.\n");
        return 1;
    }

    // 2. Warmup (Wake up GPU)
    printf("Warming up GPU...\n");
    EVP_PKEY *pkey = NULL;
    for (int i = 0; i < 10; i++) {
        EVP_PKEY_keygen(ctx, &pkey);
        EVP_PKEY_free(pkey);
        pkey = NULL;
    }

    // 3. Benchmark Loop
    int iterations = 1000; // Adjust this if it's too fast/slow
    printf("Running %d iterations...\n", iterations);
    
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        if (EVP_PKEY_keygen(ctx, &pkey) <= 0) {
            fprintf(stderr, "Error at iteration %d\n", i);
            ERR_print_errors_fp(stderr);
            break;
        }
        EVP_PKEY_free(pkey);
        pkey = NULL;
    }
    
    clock_t end = clock();
    
    // 4. Report
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n--- Results ---\n");
    printf("Total Time: %.2f seconds\n", time_spent);
    printf("Ops/Sec:    %.2f\n", iterations / time_spent);

    EVP_PKEY_CTX_free(ctx);
    return 0;
}
