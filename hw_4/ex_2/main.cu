
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
  srand(time(NULL));

  int inputLength;
  int S_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  Timer timer;

  //@@ Insert code below to read in inputLength from args

  if (argc != 3)
  {
    printf("Input lenght and segment size must be specified as the cli arguments.\n");
    printf("Ex: run.o [inputLength] [S_seg].\n");
    exit(-1);
  }

  inputLength = atoi(argv[1]);
  S_seg = atoi(argv[2]);

  printf("The input length is %d, segment size is %d\n", inputLength, S_seg);

  //@@ Create cuda streams
  cudaStream_t streams[S_seg];
  int streamSize = inputLength / S_seg;
  int streamBytes = streamSize * sizeof(DataType);

  for (size_t i = 0; i < S_seg; i++)
  {
    cudaStreamCreate(&streams[i]);
  }

  //@@ Insert code below to allocate PINNED Host memory for input and output

  cudaHostAlloc((void **)&hostInput1, sizeof(DataType) * inputLength, cudaHostAllocDefault);
  cudaHostAlloc((void **)&hostInput2, sizeof(DataType) * inputLength, cudaHostAllocDefault);
  cudaHostAlloc((void **)&hostOutput, sizeof(DataType) * inputLength, cudaHostAllocDefault);

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

  dim3 grid((streamSize + TPB - 1) / TPB);
  dim3 block(TPB);

  for (size_t i = 0; i < S_seg; i++)
  {
    int offset = streamSize * i;
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);

    vecAdd<<<grid, block, 0, streams[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);

    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
  }

  cudaDeviceSynchronize();
  timer.stop("Total runtime");

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

  for (size_t i = 0; i < S_seg; i++)
  {
    cudaStreamDestroy(streams[i]);
  }

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  free(resultRef);

  return 0;
}
