#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

int cupqc_shim_keygen_768(unsigned char *pk, unsigned char *sk) { return 0; }
void cupqc_encaps_mlkem768_batch(int count, unsigned char **pk, unsigned char **rnd, unsigned char **ct, unsigned char **ss) {}

int main() {
    printf("Benchmarking Single-Core CPU ML-KEM-768...\n");
    FILE *keyfile = fopen("mlkem_key.pem", "r");
    if(!keyfile) return 1;
    EVP_PKEY *pkey = PEM_read_PrivateKey(keyfile, NULL, NULL, NULL);
    fclose(keyfile);

    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(pkey, NULL);
    if(EVP_PKEY_encapsulate_init(ctx, NULL) <= 0) return 1;

    unsigned char *secret = malloc(32);
    unsigned char *ciphertext = malloc(1088);
    size_t s_len, c_len;
    int iters = 10000;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for(int i = 0; i < iters; i++) {
        s_len = 32; c_len = 1088;
        EVP_PKEY_encapsulate(ctx, ciphertext, &c_len, secret, &s_len);
    }

    gettimeofday(&end, NULL);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    double single_core_ops = iters / time_spent;
    printf("\n--- CPU Baseline Results ---\n");
    printf("Total Time: %.2f seconds\n", time_spent);
    printf("1-Core Ops/Sec: %.2f\n", single_core_ops);
    printf("Theoretical 4-Core Max: %.2f\n", single_core_ops * 4);

    /* FIX: Methodological consistency applied. Bypassing global atexit() */
    _exit(0); 
}
