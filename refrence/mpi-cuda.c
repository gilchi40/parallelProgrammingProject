#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern void runCudaLand(int rank, int size, int *local_in, int *local_out, int localN, int RADIUS);

void stencil_serial(int *in, int *out, int N, int RADIUS)
{
  for (int i = 0; i < N; i++)
  {
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
    {
      if (i + offset >= 0 && i + offset < N)
        out[i] += in[i + offset];
    }
  }
}

typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do
  {
    __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__("mftb %0" : "=r"(tbl));
    __asm__ __volatile__("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

int main(int argc, char **argv)
{
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from CPU land on processor %s, rank %d"
         " out of %d processors\n",
         processor_name, world_rank, world_size);

  // check for correct number of arguments
  if (argc != 3)
  {
    if (world_rank == 0)
    {
      fprintf(stderr, "Usage: %s NumElements HaloSize\n", argv[0]);
      fprintf(stderr, "  NumElements : exponent for 2^N elements\n");
      fprintf(stderr, "  HaloSize    : stencil radius\n");
    }
    MPI_Finalize();
    return 1;
  }

  int exp = atoi(argv[1]);
  int N = 1 << exp; // total number of elements
  int RADIUS = atoi(argv[2]);

  // Determine per-rank core sizes (handling remainder)
  int base = N / world_size;
  int rem = N % world_size;
  int *sendcounts = NULL;
  int *displs = NULL;
  int localN; // core elements for this rank

  // Rank 0 allocates and initialises sendcounts and displs
  if (world_rank == 0)
  {
    sendcounts = (int *)malloc(world_size * sizeof(int));
    displs = (int *)malloc(world_size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < world_size; i++)
    {
      sendcounts[i] = base + (i < rem ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  // Scatter the core sizes to all ranks (so each knows its localN)
  MPI_Scatter(sendcounts, 1, MPI_INT, &localN, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate local arrays with halo space
  int *local_in = (int *)malloc((localN + 2 * RADIUS) * sizeof(int));
  int *local_out = (int *)malloc(localN * sizeof(int));

  // Rank 0 allocates and initialises the full input array
  int *full_in = NULL;
  int *full_out = NULL;
  if (world_rank == 0)
  {
    full_in = (int *)malloc(N * sizeof(int));
    full_out = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
      full_in[i] = 1; // simple pattern
  }

  // Barrier after memory setup (good practice)
  MPI_Barrier(MPI_COMM_WORLD);

  // Start timing on rank 0
  long long start = 0;
  if (world_rank == 0)
    start = getticks();

  // Scatter core data to all ranks (into the middle of local_in)
  MPI_Scatterv(full_in, sendcounts, displs, MPI_INT,
               local_in + RADIUS, localN, MPI_INT,
               0, MPI_COMM_WORLD);

  // Halo exchange using non‑blocking communication
  // use wrap around to fix edge cases :

  // Halo exchange using non‑blocking communication
  MPI_Request requests[4];
  int nreq = 0;

  // Left neighbour (rank-1)
  if (world_rank > 0)
  {
    // Receive left halo into local_in[0 .. RADIUS-1]
    MPI_Irecv(local_in, RADIUS, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, &requests[nreq++]);
    // Send my left core boundary to left neighbour (tag 1)
    MPI_Isend(local_in + RADIUS, RADIUS, MPI_INT, world_rank - 1, 1, MPI_COMM_WORLD, &requests[nreq++]);
  }

  // Right neighbour (rank+1)
  if (world_rank < world_size - 1)
  {
    // Receive right halo into local_in[RADIUS+localN .. 2*RADIUS+localN-1]
    MPI_Irecv(local_in + RADIUS + localN, RADIUS, MPI_INT, world_rank + 1, 1, MPI_COMM_WORLD, &requests[nreq++]);
    // Send my right core boundary to right neighbour (tag 0)
    MPI_Isend(local_in + RADIUS + localN - RADIUS, RADIUS, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, &requests[nreq++]);
  }

  // Wait for all halo transfers to complete
  MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

  if (world_rank == 0)
  {
    // Left halo (global domain boundary) → zeros
    for (int i = 0; i < RADIUS; i++)
      local_in[i] = 0;
  }
  if (world_rank == world_size - 1)
  {
    // Right halo (global domain boundary) → zeros
    for (int i = 0; i < RADIUS; i++)
      local_in[RADIUS + localN + i] = 0;
  }

  runCudaLand(world_rank, world_size, local_in, local_out, localN, RADIUS);

  // Gather all local results back to rank 0
  MPI_Gatherv(local_out, localN, MPI_INT,
              full_out, sendcounts, displs, MPI_INT,
              0, MPI_COMM_WORLD);

  if (world_rank == 0)
  {
    // Stop timing on rank 0
    long long end = getticks();
    double seconds = (end - start) / 1e9;
    printf("Parallel time (including scatter/gather): %f seconds. NumElements: %d. HaloSize: %d\n", seconds, exp, RADIUS);

    // Verification on rank 0 (serial computation and comparison)
    int *serial_out = (int *)malloc(N * sizeof(int));
    double t_ser = MPI_Wtime();
    stencil_serial(full_in, serial_out, N, RADIUS);
    t_ser = MPI_Wtime() - t_ser;
    printf("Serial time: %f seconds\n", t_ser);

    int correct = 1;
    for (int i = 0; i < N; i++)
    {
      if (full_out[i] != serial_out[i])
      {
        correct = 0;
        break;
      }
    }
    if (correct)
      printf("Result verified: parallel output matches serial.\n");
    else
      printf("ERROR: parallel output does NOT match serial!\n");

    // Print first few results for visual check
    printf("First 10 output values: ");
    for (int i = 0; i < 10 && i < N; i++)
      printf("%d ", full_out[i]);
    printf("\n");

    free(serial_out);
    free(full_in);
    free(full_out);
    free(sendcounts);
    free(displs);
  }

  free(local_in);
  free(local_out);

  // Finalize the MPI environment.
  MPI_Finalize();
}
