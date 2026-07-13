/* benchmark_cpu.c - Multi-Threaded CPU Benchmark (Thread-Isolated) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

void *cpu_worker(void *arg) {
    int iters = *(int *)arg;

    FILE *keyfile = fopen("mlkem_key.pem", "r");
    EVP_PKEY *local_pkey = PEM_read_PrivateKey(keyfile, NULL, NULL, NULL);
    fclose(keyfile);

    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(local_pkey, NULL);
    EVP_PKEY_encapsulate_init(ctx, NULL);

    unsigned char *secret = malloc(32);
    unsigned char *ciphertext = malloc(1088);
    size_t s_len, c_len;

    for (int i = 0; i < iters; i++) {
        s_len = 32; c_len = 1088;
        int rc = EVP_PKEY_encapsulate(ctx, ciphertext, &c_len, secret, &s_len);
        if (rc <= 0) {
            fprintf(stderr, "[FATAL] encapsulate failed at iteration %d\n", i);
            break;
        }
    }

    free(secret); free(ciphertext);
    EVP_PKEY_CTX_free(ctx);
    EVP_PKEY_free(local_pkey);
    return NULL;
}

int main(int argc, char **argv) {
    int num_threads = 4;
    long total_iters = 100000;

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) total_iters = atol(argv[2]);

    int iters_per_thread = (int)(total_iters / num_threads);

    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
    int *iters_arg = malloc(sizeof(int) * num_threads);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        iters_arg[i] = iters_per_thread;
        pthread_create(&threads[i], NULL, cpu_worker, &iters_arg[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    long total_completed = (long)num_threads * iters_per_thread;

    printf("\n--- CPU Baseline Results ---\n");
    printf("Threads: %d | Total Ops: %ld\n", num_threads, total_completed);
    printf("Ops/Sec: %.2f\n", (double)total_completed / time_spent);

    free(threads);
    free(iters_arg);
    return 0;
}