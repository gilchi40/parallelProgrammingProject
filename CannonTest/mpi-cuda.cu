#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* Tile width for shared-memory tiling */
#define TILE_WIDTH 16

extern "C"
{
  /*
   * Cannon's algorithm calls this q times per rank.
   * A_block, B_block : [blockN x blockN]  local tiles for this step
   * C_block          : [blockN x blockN]  accumulated output (NOT zeroed here)
   * blockN           : block dimension
   * N                : passed as blockN here (blocks are square)
   */
  void runCudaLand(int rank, int size,
                   int *A_block, int *B_block, int *C_block,
                   int blockN, int N);
}

/*
 * Tiled matrix-multiply kernel — ACCUMULATES into out (out += A*B).
 * All three matrices are [blockN x blockN] row-major.
 */
__global__ void matmul_tiled_accum(const int *A, const int *B, int *C, int blockN)
{
  __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  int sum = 0;
  int numTiles = (blockN + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int t = 0; t < numTiles; t++)
  {
    int a_col = t * TILE_WIDTH + threadIdx.x;
    tileA[threadIdx.y][threadIdx.x] =
        (row < blockN && a_col < blockN) ? A[row * blockN + a_col] : 0;

    int b_row = t * TILE_WIDTH + threadIdx.y;
    tileB[threadIdx.y][threadIdx.x] =
        (b_row < blockN && col < blockN) ? B[b_row * blockN + col] : 0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

    __syncthreads();
  }

  /* Accumulate — C is NOT reset between Cannon steps */
  if (row < blockN && col < blockN){
    C[row * blockN + col] += sum;
  }
}

void runCudaLand(int rank, int size,
                 int *A_block, int *B_block, int *C_block,
                 int blockN, int N)
{
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

  size_t bytes = (size_t)blockN * blockN * sizeof(int);

  int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  if ((err = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "Rank %d: cudaMalloc failed: %s\n",
            rank, cudaGetErrorString(err));
    return;
  }

  cudaMemcpy(d_A, A_block, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_block, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C_block, bytes, cudaMemcpyHostToDevice); /* load existing C */

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((blockN + TILE_WIDTH - 1) / TILE_WIDTH,
               (blockN + TILE_WIDTH - 1) / TILE_WIDTH);

  matmul_tiled_accum<<<gridDim, blockDim>>>(d_A, d_B, d_C, blockN);

  cudaDeviceSynchronize();
  if ((err = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "Rank %d: kernel error: %s\n",
            rank, cudaGetErrorString(err));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return;
  }
  /* Write accumulated C back to host */
  cudaMemcpy(C_block, d_C, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

}
