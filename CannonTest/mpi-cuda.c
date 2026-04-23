#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Cannon's Algorithm — MPI + CUDA matrix multiply
 *
 * Requires world_size = q*q (perfect square) MPI ranks.
 * Each rank owns a (blockN x blockN) block of A, B, and C,
 * where blockN = N / q.
 *
 * runCudaLand multiplies its local A block by its local B block
 * and ACCUMULATES into C (does NOT zero C first).
 */
extern void runCudaLand(int rank, int size,
                        int *A_block, int *B_block, int *C_block,
                        int blockN, int N);

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

/*
 * Scatter a full N x N matrix (row-major on rank 0) into
 * q x q blocks of size blockN x blockN, one block per rank.
 */
static void scatter_blocks(int *full, int *block,
                           int N, int blockN, int q,
                           MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  int *sendbuf = NULL;
  if (rank == 0)
  {
    sendbuf = (int *)malloc((size_t)N * N * sizeof(int));
    for (int pr = 0; pr < q; pr++)
      for (int pc = 0; pc < q; pc++)
        for (int i = 0; i < blockN; i++)
          for (int j = 0; j < blockN; j++)
          {
            int dst = ((pr * q + pc) * blockN + i) * blockN + j;
            int src = (pr * blockN + i) * N + (pc * blockN + j);
            sendbuf[dst] = full[src];
          }
  }

  MPI_Scatter(sendbuf, blockN * blockN, MPI_INT,
              block,   blockN * blockN, MPI_INT,
              0, comm);

  if (rank == 0) free(sendbuf);
}

/*
 * Gather q x q contiguous blocks back into a full N x N matrix on rank 0.
 */
static void gather_blocks(int *block, int *full,
                          int N, int blockN, int q,
                          MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  int *recvbuf = NULL;
  if (rank == 0)
    recvbuf = (int *)malloc((size_t)N * N * sizeof(int));

  MPI_Gather(block,   blockN * blockN, MPI_INT,
             recvbuf, blockN * blockN, MPI_INT,
             0, comm);

  if (rank == 0)
  {
    for (int pr = 0; pr < q; pr++)
      for (int pc = 0; pc < q; pc++)
        for (int i = 0; i < blockN; i++)
          for (int j = 0; j < blockN; j++)
          {
            int src = ((pr * q + pc) * blockN + i) * blockN + j;
            int dst = (pr * blockN + i) * N + (pc * blockN + j);
            full[dst] = recvbuf[src];
          }
    free(recvbuf);
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

  /* Cannon's requires a square process grid */
  int q = (int)round(sqrt((double)world_size));
  if (q * q != world_size)
  {
    if (world_rank == 0)
      fprintf(stderr,
              "Cannon's algorithm requires a square number of MPI ranks "
              "(got %d).\n", world_size);
    MPI_Finalize();
    return 1;
  }

  if (argc != 2)
  {
    if (world_rank == 0)
      fprintf(stderr, "Usage: %s <exp>  (matrix dimension 2^exp x 2^exp)\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int exp    = atoi(argv[1]);
  int N      = 1 << exp; /* 2^exp */
  int blockN = N / q;    /* block dimension per rank */

  if (N % q != 0)
  {
    if (world_rank == 0)
      fprintf(stderr, "N (%d) must be divisible by q (%d).\n", N, q);
    MPI_Finalize();
    return 1;
  }

  /* ------------------------------------------------------------------ */
  /* Create a q x q Cartesian communicator with wrap-around             */
  /* ------------------------------------------------------------------ */
  int dims[2]    = {q, q};
  int periods[2] = {1, 1};
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

  int cart_rank;
  int coords[2];
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
  int my_row = coords[0];
  int my_col = coords[1];

  /* Row and column sub-communicators for shifts */
  MPI_Comm row_comm, col_comm;
  int remain_row[2] = {0, 1};
  int remain_col[2] = {1, 0};
  MPI_Cart_sub(cart_comm, remain_row, &row_comm);
  MPI_Cart_sub(cart_comm, remain_col, &col_comm);

  /* ------------------------------------------------------------------ */
  /* Allocate local blocks                                               */
  /* ------------------------------------------------------------------ */
  size_t block_bytes = (size_t)blockN * blockN * sizeof(int);
  int *A_block = (int *)malloc(block_bytes);
  int *B_block = (int *)malloc(block_bytes);
  int *C_block = (int *)calloc(blockN * blockN, sizeof(int)); /* zeroed */

  /* ------------------------------------------------------------------ */
  /* Rank 0: build full A and B (all ones for testing)                  */
  /* ------------------------------------------------------------------ */
  int *full_A = NULL, *full_B = NULL, *full_C = NULL;
  if (world_rank == 0)
  {
    full_A = (int *)malloc((size_t)N * N * sizeof(int));
    full_B = (int *)malloc((size_t)N * N * sizeof(int));
    full_C = (int *)malloc((size_t)N * N * sizeof(int));
    for (int i = 0; i < N * N; i++) full_A[i] = 1;
    for (int i = 0; i < N * N; i++) full_B[i] = 1;
  }

  /* ------------------------------------------------------------------ */
  /* Scatter blocks of A and B to all ranks                             */
  /* ------------------------------------------------------------------ */
  scatter_blocks(full_A, A_block, N, blockN, q, MPI_COMM_WORLD);
  scatter_blocks(full_B, B_block, N, blockN, q, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

  /* ------------------------------------------------------------------ */
  /* Cannon's initial skew                                               */
  /* Shift A left by my_row steps, B up by my_col steps                */
  /* ------------------------------------------------------------------ */
  {
    int src, dst;
    MPI_Cart_shift(row_comm, 0, -my_row, &src, &dst);
    MPI_Sendrecv_replace(A_block, blockN * blockN, MPI_INT,
                         dst, 0, src, 0, row_comm, MPI_STATUS_IGNORE);
  }
  {
    int src, dst;
    MPI_Cart_shift(col_comm, 0, -my_col, &src, &dst);
    MPI_Sendrecv_replace(B_block, blockN * blockN, MPI_INT,
                         dst, 0, src, 0, col_comm, MPI_STATUS_IGNORE);
  }

  /* ------------------------------------------------------------------ */
  /* Cannon's main loop: q steps                                         */
  /* Each step: C_block += A_block * B_block, shift A left, shift B up */
  /* ------------------------------------------------------------------ */
  for (int step = 0; step < q; step++)
  {
    runCudaLand(cart_rank, world_size,
                A_block, B_block, C_block,
                blockN, blockN);

    /* Shift A left by 1 */
    {
      int src, dst;
      MPI_Cart_shift(row_comm, 0, -1, &src, &dst);
      MPI_Sendrecv_replace(A_block, blockN * blockN, MPI_INT,
                           dst, 0, src, 0, row_comm, MPI_STATUS_IGNORE);
    }
    /* Shift B up by 1 */
    {
      int src, dst;
      MPI_Cart_shift(col_comm, 0, -1, &src, &dst);
      MPI_Sendrecv_replace(B_block, blockN * blockN, MPI_INT,
                           dst, 0, src, 0, col_comm, MPI_STATUS_IGNORE);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Gather C blocks back to rank 0                                      */
  /* ------------------------------------------------------------------ */
  gather_blocks(C_block, full_C, N, blockN, q, MPI_COMM_WORLD);

  double t_end = MPI_Wtime();

  /* ------------------------------------------------------------------ */
  /* Verification on rank 0                                              */
  /* ------------------------------------------------------------------ */
  if (world_rank == 0)
  {
    printf("Cannon parallel time: %.6f s  (exp=%d, N=%d, q=%d)\n",
           t_end - t_start, exp, N, q);

    int *serial_C = (int *)malloc((size_t)N * N * sizeof(int));
    double t_ser = MPI_Wtime();
    matmul_serial(full_A, full_B, serial_C, N);
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
    free(full_B);
    free(full_C);
  }

  free(A_block);
  free(B_block);
  free(C_block);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();
  return 0;
}
