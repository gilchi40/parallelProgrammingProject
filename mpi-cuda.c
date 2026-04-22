#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * runCudaLand:
 *   rank, size  – MPI rank / world size
 *   A_rows      – pointer to this rank's rows of A  [localRows x N]
 *   B           – full matrix B                     [N x N]
 *   C_rows      – output: this rank's rows of C     [localRows x N]
 *   localRows   – number of rows owned by this rank
 *   N           – matrix dimension
 */
extern void runCudaLand(int rank, int size,
                        int *A_rows, int *B, int *C_rows,
                        int localRows, int N);

/* Serial matrix multiply for verification (rank-0 only) */
static void matmul_serial(const int *A, const int *B, int *C, int N)
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    {
      long long sum = 0;
      for (int k = 0; k < N; k++)
        sum += (long long)A[i * N + k] * B[k * N + j];
      C[i * N + j] = (int)sum;
    }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello world from processor %s, rank %d out of %d\n",
         processor_name, world_rank, world_size);

  if (argc != 2)
  {
    if (world_rank == 0)
      fprintf(stderr, "Usage: %s <N>  (matrix dimension N x N)\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int exp = atoi(argv[1]);
  int N = 1 << exp; // total number of elements

  if (N <= 0)
  {
    if (world_rank == 0)
      fprintf(stderr, "N must be a positive integer.\n");
    MPI_Finalize();
    return 1;
  }

  /* ------------------------------------------------------------------ */
  /* Distribute rows of A across ranks                                    */
  /* ------------------------------------------------------------------ */
  int base = N / world_size;
  int rem  = N % world_size;

  int *sendcounts_rows = NULL; /* number of rows per rank          */
  int *sendcounts      = NULL; /* number of ints per rank (rows*N) */
  int *displs          = NULL; /* displacement in ints             */

  if (world_rank == 0)
  {
    sendcounts_rows = (int *)malloc(world_size * sizeof(int));
    sendcounts      = (int *)malloc(world_size * sizeof(int));
    displs          = (int *)malloc(world_size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < world_size; i++)
    {
      sendcounts_rows[i] = base + (i < rem ? 1 : 0);
      sendcounts[i]      = sendcounts_rows[i] * N;
      displs[i]          = offset;
      offset            += sendcounts[i];
    }
  }

  /* Each rank learns its own row count */
  int localRows;
  MPI_Scatter(sendcounts_rows, 1, MPI_INT,
              &localRows,      1, MPI_INT,
              0, MPI_COMM_WORLD);

  /* ------------------------------------------------------------------ */
  /* Allocate matrices                                                    */
  /* ------------------------------------------------------------------ */
  int *A_rows = (int *)malloc((size_t)localRows * N * sizeof(int));
  int *B      = (int *)malloc((size_t)N         * N * sizeof(int));
  int *C_rows = (int *)malloc((size_t)localRows * N * sizeof(int));
  memset(C_rows, 0, (size_t)localRows * N * sizeof(int));

  int *full_A = NULL, *full_C = NULL;
  if (world_rank == 0)
  {
    full_A = (int *)malloc((size_t)N * N * sizeof(int));
    full_C = (int *)malloc((size_t)N * N * sizeof(int));

    /* Fill A and B with 1s for testing */
    for (int i = 0; i < N * N; i++)
      full_A[i] = 1;
    for (int i = 0; i < N * N; i++)
      B[i] = 1;
  }

  /* Broadcast B to every rank */
  MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

  /* Scatter rows of A */
  MPI_Scatterv(full_A,  sendcounts, displs, MPI_INT,
               A_rows, localRows * N,        MPI_INT,
               0, MPI_COMM_WORLD);

  /* ------------------------------------------------------------------ */
  /* GPU kernel (each rank processes its local rows)                     */
  /* ------------------------------------------------------------------ */
  runCudaLand(world_rank, world_size,
              A_rows, B, C_rows,
              localRows, N);

  /* ------------------------------------------------------------------ */
  /* Gather results back to rank 0                                        */
  /* ------------------------------------------------------------------ */
  MPI_Gatherv(C_rows,  localRows * N, MPI_INT,
              full_C, sendcounts, displs, MPI_INT,
              0, MPI_COMM_WORLD);

  double t_end = MPI_Wtime();

  /* ------------------------------------------------------------------ */
  /* Verification (rank 0 only)                                           */
  /* ------------------------------------------------------------------ */
  if (world_rank == 0 && false)
  {
    printf("Parallel wall time (scatter+compute+gather): %.6f s  (N=%d)\n",
           t_end - t_start, N);

    int *serial_C = (int *)malloc((size_t)N * N * sizeof(int));
    double t_ser = MPI_Wtime();
    matmul_serial(full_A, B, serial_C, N);
    t_ser = MPI_Wtime() - t_ser;
    printf("Serial time: %.6f s\n", t_ser);

    int correct = 1;
    for (int i = 0; i < N * N; i++)
    {
      if (full_C[i] != serial_C[i])
      {
        printf("MISMATCH at element %d: got %d, expected %d\n",
               i, full_C[i], serial_C[i]);
        correct = 0;
        break;
      }
    }
    printf("%s\n", correct
             ? "Result verified: parallel output matches serial."
             : "ERROR: parallel output does NOT match serial!");

    /* For an all-ones matrix, every element should equal N */
    printf("Expected value per element: %d  |  C[0][0] = %d\n", N, full_C[0]);

    printf("Top-left 4x4 corner of C:\n");
    for (int i = 0; i < 4 && i < N; i++)
    {
      for (int j = 0; j < 4 && j < N; j++)
        printf("%6d ", full_C[i * N + j]);
      printf("\n");
    }

    free(serial_C);
    free(full_A);
    free(full_C);
    free(sendcounts_rows);
    free(sendcounts);
    free(displs);
  }

  free(A_rows);
  free(B);
  free(C_rows);

  MPI_Finalize();
  return 0;
}
