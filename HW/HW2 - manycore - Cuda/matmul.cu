#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_ITERATIONS 10
#define NUM_TESTS 20
#define MAX_ELEM 100.0
#define MIN_ELEM -100.0
#define TILE 16

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_tiled_kernel(const double *A, const double *B, double *C,
                                    int M, int K, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;

    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void populate_matrix(double *mat, int rows, int cols, double min_val, double max_val) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = min_val + ((double)rand() / RAND_MAX) * (max_val - min_val);
}

void generate_matrices(double *A, double *B, int M, int K, int N,
                       double min_val, double max_val) {
    populate_matrix(A, M, K, min_val, max_val);
    populate_matrix(B, K, N, min_val, max_val);
}

double matmul_cuda(double *A, double *B, double *C, int M, int K, int N) {
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;

    size_t sizeA = (size_t)M * K * sizeof(double);
    size_t sizeB = (size_t)K * N * sizeof(double);
    size_t sizeC = (size_t)M * N * sizeof(double);

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return ms / 1000.0; // seconds
}

void run_simulation(double *times_cuda, int index, int M, int K, int N) {
    double *A = (double *)malloc((size_t)M * K * sizeof(double));
    double *B = (double *)malloc((size_t)K * N * sizeof(double));
    double *C = (double *)calloc((size_t)M * N, sizeof(double));

    double total_time = 0.0;

    for (int i = 0; i < NUM_TESTS; i++) {
        generate_matrices(A, B, M, K, N, MIN_ELEM, MAX_ELEM);
        total_time += matmul_cuda(A, B, C, M, K, N);
    }

    times_cuda[index] = total_time / (double)NUM_TESTS;

    free(A);
    free(B);
    free(C);
}

int main() {
    srand((unsigned)time(NULL));

    int M_sizes[NUM_ITERATIONS] = {100, 200, 300, 400, 512, 600, 700, 800, 1024, 2048};
    int K_sizes[NUM_ITERATIONS] = {150, 250, 350, 450, 512, 650, 750, 850, 1024, 2048};
    int N_sizes[NUM_ITERATIONS] = {200, 300, 400, 500, 512, 700, 800, 900, 1024, 2048};

    double times_cuda[NUM_ITERATIONS];

    FILE *csv = fopen("results_cuda.csv", "w");
    if (!csv) {
        fprintf(stderr, "Error opening CSV file!\n");
        return 1;
    }

    fprintf(csv, "time,M,K,N\n");

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        printf("Test %d: M=%d, K=%d, N=%d\n", i + 1, M_sizes[i], K_sizes[i], N_sizes[i]);

        run_simulation(times_cuda, i, M_sizes[i], K_sizes[i], N_sizes[i]);

        printf("Execution time: %.6f seconds\n\n", times_cuda[i]);
        fprintf(csv, "%.6f,%d,%d,%d\n",
                times_cuda[i], M_sizes[i], K_sizes[i], N_sizes[i]);
    }

    printf("=== RESULTS ===\n");
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        printf("Test %d (M=%d, K=%d, N=%d): %.6f s\n",
               i + 1, M_sizes[i], K_sizes[i], N_sizes[i], times_cuda[i]);
    }

    fclose(csv);
    return 0;
}