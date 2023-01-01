
#include <stdio.h>
#include <sys/time.h>
#include "../common/timer.h"

#define DataType float
#define TPB 32

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                     int numAColumns, int numBRows, int numBColumns)
{
  //@@ Insert code to implement matrix multiplication here
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= numARows || j >= numBColumns)
  {
    return;
  }

  DataType tempSum = 0;

  for (int l = 0; l < numBRows; l++)
  {
    tempSum += A[i * numAColumns + l] * B[l * numBColumns + j];
  }

  C[i * numBColumns + j] = tempSum;
}

void fillMatrixWithRandom(DataType *matrix, int *numRows, int *numColumns)
{
  for (int i = 0; i < *numRows; i++)
  {
    for (int j = 0; j < *numColumns; j++)
    {
      matrix[i * (*numColumns) + j] = ((DataType)rand() / (DataType)(RAND_MAX));
    }
  }
}

int main(int argc, char **argv)
{
  srand ( time(NULL) );

  DataType *hostA;     // The A matrix
  DataType *hostB;     // The B matrix
  DataType *hostC;     // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;
  Timer timerTotal;
  Timer timer;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  if (argc != 4)
  {
    printf("Rows and columns of vector A and B must be specified as the cli arguments.\n");
    printf("Ex: ex2.o [numARows] [numAColumns] [numBColumns].\n");
    exit(-1);
  }

  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[2]);
  numBColumns = atoi(argv[3]);

  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output

  timer.start();
  cudaMallocHost((void**)&hostA,sizeof(DataType) * numARows * numAColumns);
  cudaMallocHost((void**)&hostB,sizeof(DataType) * numBRows * numBColumns);
  cudaMallocHost((void**)&hostC,sizeof(DataType) * numCRows * numCColumns);
  timer.stop("Memory allocation (Host)");
  resultRef = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  fillMatrixWithRandom(hostA, &numARows, &numAColumns);
  fillMatrixWithRandom(hostB, &numBRows, &numBColumns);

  for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      resultRef[i * numBColumns + j] = 0;
      for (int l = 0; l < numBRows; l++)
      {
        resultRef[i * numBColumns + j] += hostA[i * numAColumns + l] * hostB[l * numBColumns + j];
      }
    }
  }

  timerTotal.start();
  timer.start();
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, sizeof(DataType) * numARows * numAColumns);
  cudaMalloc(&deviceB, sizeof(DataType) * numBRows * numBColumns);
  cudaMalloc(&deviceC, sizeof(DataType) * numCRows * numCColumns);
  timer.stop("Memory allocation (Device)");

  //@@ Insert code to below to Copy memory to the GPU here
  timer.start();
  cudaMemcpy(deviceA, hostA, sizeof(DataType) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(DataType) * numBRows * numBColumns, cudaMemcpyHostToDevice);
  timer.stop("Host to device copy");

  //@@ Initialize the grid and block dimensions here
  dim3 grid((numBColumns + TPB - 1) / TPB, (numARows + TPB - 1) / TPB);
  dim3 block(TPB, TPB);

  timer.start();
  //@@ Launch the GPU Kernel here
  gemm<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

  cudaDeviceSynchronize();
  timer.stop("kernel");

  //@@ Copy the GPU memory back to the CPU here
  timer.start();
  cudaMemcpy(hostC, deviceC, sizeof(DataType) * numCRows * numCColumns, cudaMemcpyDeviceToHost);
  timer.stop("Device to Host copy");
  timerTotal.stop("Total runtime");

  //@@ Insert code below to compare the output with the reference
  float meanError = 0;
  for (int i = 0; i < numCRows; i++)
  {
    for (int j = 0; j < numCColumns; j++){
      meanError += fabs(resultRef[i * numCColumns + j] - hostC[i * numCColumns + j]);
    }
  }

  meanError /= numCRows * numCColumns;

  printf("Mean error is %f \n", meanError);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  cudaFreeHost(hostA);
  cudaFreeHost(hostB);
  cudaFreeHost(hostC);
  free(resultRef);

  return 0;
}
