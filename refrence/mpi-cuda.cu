#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
  void runCudaLand(int rank, int size, int *local_in, int *local_out, int localN, int RADIUS);
}

__global__ void stencil_1d(int *in, int *out, int N, int localN, int RADIUS);

void runCudaLand(int rank, int size, int *local_in, int *local_out, int localN, int RADIUS)
{

  // remainder of the computation would go here (e.g., stencil operation on local_in to produce local_out)
  printf("MPI rank %d: leaving CPU land \n", rank);

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    fprintf(stderr, "MPI rank %d: No CUDA-capable device found.\n", rank);
    return;
  }

  int device = rank % deviceCount;
  cudaSetDevice(device);
  cudaError_t err;

  size_t in_size = (localN + 2 * RADIUS) * sizeof(int);
  size_t out_size = localN * sizeof(int);

  int *d_in = nullptr, *d_out = nullptr;
  cudaMallocManaged(&d_in, in_size);
  cudaMallocManaged(&d_out, out_size);

  if ((err = cudaGetLastError()) != cudaSuccess)
  {
    printf("Rank %d: cudaMallocManaged failed: %s\n", rank, cudaGetErrorString(err));
    return;
  }

  // Copy data from host pointers to managed memory
  memcpy(d_in, local_in, in_size);

  // Prefetch input and output to the selected GPU
  cudaMemPrefetchAsync(d_in, in_size, device, 0);
  cudaMemPrefetchAsync(d_out, out_size, device, 0);
  cudaDeviceSynchronize(); // ensure prefetch completes before kernel launch

  // Hello_kernel<<<128, 128>>>(rank);
  stencil_1d<<<16384, 1024>>>(d_in, d_out, localN, localN + 2 * RADIUS, RADIUS);
  // Wait for kernel to finish and check for errors
  cudaDeviceSynchronize();
  if ((err = cudaGetLastError()) != cudaSuccess)
  {
    printf("Rank %d: Kernel launch failed: %s\n", rank, cudaGetErrorString(err));
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }

  // Prefetch result back to host (so CPU can read it)
  cudaMemPrefetchAsync(d_out, out_size, cudaCpuDeviceId, 0);
  cudaDeviceSynchronize();

  // Copy result back to the original host output buffer
  memcpy(local_out, d_out, out_size);

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);

  printf("MPI rank %d: re-entering CPU land \n", rank);
}

__global__ void stencil_1d(int *in, int *out, int outN, int inN, int RADIUS)
{

  int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int start = tid + RADIUS;

  for (int base = start; base < inN - RADIUS; base += stride)
  {
    // halos are aready loaded into in
    // in is radius halo elements, N elments and radius halo elements

    int out_idx = base - RADIUS;

    // Apply stencil only if inside valid output range
    if (out_idx < outN)
    {
      int result = 0;

      // Loop over stencil
      for (int offset = -RADIUS; offset <= RADIUS; offset++)
      {
        result += in[base + offset];
      }
      out[out_idx] = result;
    }
  }
}
