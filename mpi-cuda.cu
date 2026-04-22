#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* Tile width for shared-memory tiling */
#define TILE_WIDTH 16

extern "C"
{
  void runCudaLand(int rank, int size,
                   int *A_rows, int *B, int *C_rows,
                   int localRows, int N);
}

/*
 * Tiled matrix-multiply kernel.
 *
 * Each thread block computes one TILE_WIDTH × TILE_WIDTH tile of C.
 *
 * Grid  : (ceil(N/TILE_WIDTH), ceil(localRows/TILE_WIDTH))
 * Block : (TILE_WIDTH, TILE_WIDTH)
 *
 * A_rows : [localRows × N]   (row-major)
 * B      : [N × N]           (row-major)
 * C_rows : [localRows × N]   (row-major)
 */
__global__ void matmul_tiled(const int *A_rows, const int *B, int *C_rows,
                              int localRows, int N)
{
  __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y; /* row in C_rows  */
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x; /* col in C_rows  */

  int sum = 0;
  int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int t = 0; t < numTiles; t++)
  {
    /* Load one tile from A (guard out-of-bound threads) */
    int a_col = t * TILE_WIDTH + threadIdx.x;
    tileA[threadIdx.y][threadIdx.x] =
        (row < localRows && a_col < N) ? A_rows[row * N + a_col] : 0;

    /* Load one tile from B */
    int b_row = t * TILE_WIDTH + threadIdx.y;
    tileB[threadIdx.y][threadIdx.x] =
        (b_row < N && col < N) ? B[b_row * N + col] : 0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

    __syncthreads();
  }

  if (row < localRows && col < N)
    C_rows[row * N + col] = sum;
}

void runCudaLand(int rank, int size,
                 int *A_rows, int *B, int *C_rows,
                 int localRows, int N)
{
  printf("MPI rank %d: entering GPU land  (localRows=%d, N=%d)\n",
         rank, localRows, N);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    fprintf(stderr, "MPI rank %d: no CUDA-capable device found.\n", rank);
    return;
  }

  int device = rank % deviceCount;
  cudaSetDevice(device);
  cudaError_t err;

  size_t A_bytes = (size_t)localRows * N * sizeof(int);
  size_t B_bytes = (size_t)N         * N * sizeof(int);
  size_t C_bytes = (size_t)localRows * N * sizeof(int);

  int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, A_bytes);
  cudaMalloc(&d_B, B_bytes);
  cudaMalloc(&d_C, C_bytes);

  if ((err = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "Rank %d: cudaMalloc failed: %s\n",
            rank, cudaGetErrorString(err));
    return;
  }

  cudaMemcpy(d_A, A_rows, A_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B,      B_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0,      C_bytes);

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((N         + TILE_WIDTH - 1) / TILE_WIDTH,
               (localRows + TILE_WIDTH - 1) / TILE_WIDTH);

  matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, localRows, N);

  cudaDeviceSynchronize();
  if ((err = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "Rank %d: kernel error: %s\n",
            rank, cudaGetErrorString(err));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return;
  }

  cudaMemcpy(C_rows, d_C, C_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  printf("MPI rank %d: leaving GPU land\n", rank);
}
