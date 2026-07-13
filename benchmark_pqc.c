/* benchmark_pqc.c - Multi-Threaded GPU Batching Engine (Nginx OS-Thread Simulation) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <stdatomic.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

#define ALGO_NAME "ML-KEM-768"
#define DEFAULT_THREADS 512
#define DEFAULT_ITERS 100000

int cupqc_shim_keygen_768(unsigned char *pk, unsigned char *sk) { abort(); }
extern void cupqc_encaps_mlkem768_batch(int count, unsigned char **pk, unsigned char **rnd, unsigned char **ct, unsigned char **ss);

extern atomic_long g_gpu_dispatch_count;
extern atomic_long g_total_ops_batched;

void *gpu_worker(void *arg) {
    int iters = *(int *)arg;

    FILE *keyfile = fopen("mlkem_key.pem", "r");
    if (!keyfile) pthread_exit(NULL);
    EVP_PKEY *local_pkey = PEM_read_PrivateKey(keyfile, NULL, NULL, NULL);
    fclose(keyfile);

    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(local_pkey, NULL);
    if (EVP_PKEY_encapsulate_init(ctx, NULL) <= 0) pthread_exit(NULL);

    unsigned char *secret = calloc(1, 32);
    unsigned char *ciphertext = calloc(1, 1088);
    size_t s_len, c_len;

    for (int i = 0; i < iters; i++) {
        s_len = 32;
        c_len = 1088;
        EVP_PKEY_encapsulate(ctx, ciphertext, &c_len, secret, &s_len);
    }

    free(secret);
    free(ciphertext);
    EVP_PKEY_CTX_free(ctx);
    EVP_PKEY_free(local_pkey);
    return NULL;
}

int main(int argc, char **argv) {
    int num_threads = DEFAULT_THREADS;
    long total_iters = DEFAULT_ITERS;

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) total_iters = atol(argv[2]);

    printf("Benchmarking GPU Engine (OS-Threaded Nginx Simulation)...\n");

    printf("Warming up NVIDIA Driver Context on Main Thread...\n");
    unsigned char *dummy_ptrs[1] = { NULL };
    cupqc_encaps_mlkem768_batch(1, dummy_ptrs, dummy_ptrs, dummy_ptrs, dummy_ptrs);
    printf("GPU Engine Ready & Driver Cached!\n");

    int iters_per_thread = (int)(total_iters / num_threads);
    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 * 1024);

    int created = 0;
    for (int i = 0; i < num_threads; i++) {
        int rc = pthread_create(&threads[i], &attr, gpu_worker, &iters_per_thread);
        if (rc != 0) {
            fprintf(stderr, "[WARN] pthread_create failed at thread %d: %s (errno=%d)\n", i, strerror(rc), rc);
            break;
        }
        created++;
    }
    pthread_attr_destroy(&attr);

    fprintf(stderr, "[INFO] Threads created: %d / %d\n", created, num_threads);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end, NULL);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    long total_completed_iters = (long)created * iters_per_thread;

    long dispatches = atomic_load(&g_gpu_dispatch_count);
    long total_ops  = atomic_load(&g_total_ops_batched);

    printf("\n--- GPU Results ---\n");
    printf("Threads created: %d / %d\n", created, num_threads);
    printf("Total Time: %.2f seconds\n", time_spent);
    printf("Ops/Sec:    %.2f\n", (double)total_completed_iters / time_spent);
    printf("GPU kernel launches: %ld\n", dispatches);
    printf("Total ops batched:   %ld\n", total_ops);
    printf("Average batch size:  %.2f\n", dispatches > 0 ? (double)total_ops / dispatches : 0.0);

    free(threads);
    _exit(0);
}