#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Number of iterations for the test
# define NUM_ITERATIONS 10
# define NUM_TESTS 20
# define MAX_ELEM 100.0
# define MIN_ELEM -100.0

// Transposes matrix B (K x N) into BT (N x K)
void transpose(double *B, double *BT, int K, int N) {
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            BT[j*K + i] = B[i*N + j];
}

// C = A * B, where A: M x K, B: K x N, C: M x N
double matmul_transposed_sequential(double *A, double *B, double *C, int M, int K, int N) {
    double start_seq = omp_get_wtime();
    double *BT = (double *)malloc(N * K * sizeof(double));
    transpose(B, BT, K, N);  // BT is N x K

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum += A[i*K + k] * BT[j*K + k]; // Both are read SEQUENTIALLY - cache friendly!
            C[i*N + j] = sum;
        }
    }

    free(BT);
    double end_seq = omp_get_wtime();
    return end_seq - start_seq; // Return time in seconds
}

// Generates a matrix with random values between min_val and max_val
void populate_matrix(double **mat, int rows, int cols, double min_val, double max_val) {
    for (int i = 0; i < rows * cols; i++)
        (*mat)[i] = min_val + ((double)rand() / RAND_MAX) * (max_val - min_val);
}

// Generates A (M x K) and B (K x N) and stores them in the provided pointers
void generate_matrices(double **A, double **B, int M, int K, int N,
                       double min_val, double max_val) {
    populate_matrix(A, M, K, min_val, max_val);
    populate_matrix(B, K, N, min_val, max_val);
}

void run_simulation(double *times_seq, int index, int M, int K, int N) {
    // Memory allocation for matrices
    double *A = (double *)malloc(M * K * sizeof(double));
    double *B = (double *)malloc(K * N * sizeof(double));
    double *C = calloc(M * N, sizeof(double));

    double time_sequential = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        generate_matrices(&A, &B, M, K, N, MIN_ELEM, MAX_ELEM);
        // Measure execution time
        time_sequential += matmul_transposed_sequential(A, B, C, M, K, N);
    }

    // Save time into array (in seconds)
    times_seq[index] = (time_sequential) / (double) NUM_TESTS;

    // Free memory
    free(A);
    free(B);
    free(C);
}

int main() {
    // Representative values for matrix dimensions
    int M_sizes[NUM_ITERATIONS] = {100, 200, 300, 400, 512, 600, 700, 800, 1024, 2048};
    int K_sizes[NUM_ITERATIONS] = {150, 250, 350, 450, 512, 650, 750, 850, 1024, 2048};
    int N_sizes[NUM_ITERATIONS] = {200, 300, 400, 500, 512, 700, 800, 900, 1024, 2048};

    // Array for storing execution times_seq
    double times_seq[NUM_ITERATIONS];
    FILE *csv = fopen("results_sequential.csv", "w");

    if (!csv) {
        fprintf(stderr, "Error opening CSV file!\n");
        return 1;
    }

    fprintf(csv, "time,M,K,N\n");

    // Run tests
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        printf("Test %d: M=%d, K=%d, N=%d\n", i + 1, M_sizes[i], K_sizes[i], N_sizes[i]);
        run_simulation(times_seq, i, M_sizes[i], K_sizes[i], N_sizes[i]);
        printf("Execution time: %.6f seconds\n\n", times_seq[i]);
        fprintf(csv, "%.6f,%d,%d,%d\n", times_seq[i], M_sizes[i], K_sizes[i], N_sizes[i]);
    }

    // Print all results
    printf("=== RESULTS ===\n");
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        printf("Test %d (M=%d, K=%d, N=%d): %.6f s\n", 
               i + 1, M_sizes[i], K_sizes[i], N_sizes[i], times_seq[i]);
    }

    fclose(csv);

    return 0;
}