
#include <stdio.h>
#include <sys/time.h>
#include "../common/timer.h"

#define DataType double
#define TPB 64

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len)
  {
    out[i] = in1[i] + in2[i];
  }
}

int main(int argc, char **argv)
{
  srand ( time(NULL) );

  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  Timer timer;

  //@@ Insert code below to read in inputLength from args

  if (argc != 2)
  {
    printf("Input length must be specified as the first argument.\n");
    exit(-1);
  }

  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output

  hostInput1 = (double *)malloc(sizeof(DataType) * inputLength);
  hostInput2 = (double *)malloc(sizeof(DataType) * inputLength);
  hostOutput = (double *)malloc(sizeof(DataType) * inputLength);
  resultRef = (double *)malloc(sizeof(DataType) * inputLength);

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

  for (int i = 0; i < inputLength; i++)
  {
    hostInput1[i] = ((float)rand() / (float)(RAND_MAX));
    hostInput2[i] = ((float)rand() / (float)(RAND_MAX));
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);

  //@@ Insert code to below to Copy memory to the GPU here

  timer.start();
  cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  timer.stop("Host to device copy");

  //@@ Initialize the 1D grid and block dimensions here
  dim3 grid((inputLength + TPB - 1) / TPB, 1, 1);
  dim3 block(TPB, 1, 1);

  //@@ Launch the GPU Kernel here
  timer.start();
  vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  timer.stop("Kernel");
  //@@ Copy the GPU memory back to the CPU here
  timer.start();
  cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);
  timer.stop("Device to Host copy");

  //@@ Insert code below to compare the output with the reference
  float meanError;

  for (int i = 0; i < inputLength; i++)
  {
    meanError += fabs(resultRef[i] - hostOutput[i]);
  }

  meanError /= inputLength;

  printf("Mean error is %f \n", meanError);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
