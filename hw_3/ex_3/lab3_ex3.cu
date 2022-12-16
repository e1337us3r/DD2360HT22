
#include <stdio.h>
#include <../common/timer.h>

#define NUM_BINS 4096
#define TPB 512

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins)
{

  //@@ Insert code below to compute histogram of input using shared memory and atomics
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= num_elements)
  {
    return;
  }

  __shared__ unsigned int sharedBins[NUM_BINS];

  for (size_t j = threadIdx.x; j <  NUM_BINS; j+=blockDim.x)
  {
    sharedBins[j] = 0;
  }
  __syncthreads();

  atomicAdd(&sharedBins[input[i]], 1);
  __syncthreads();

  for (size_t j = threadIdx.x; j <  NUM_BINS; j+=blockDim.x)
  {
    if (sharedBins[j])
    {
      atomicAdd(&bins[j], sharedBins[j]);
    }
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{
  //@@ Insert code below to clean up bins that saturate at 127
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= num_bins)
  {
    return;
  }

  if (bins[i] > 127)
  {
    bins[i] = 127;
  }
}

int main(int argc, char **argv)
{
  srand(time(NULL));

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
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
  hostInput = (unsigned int *)malloc(sizeof(unsigned int) * inputLength);
  hostBins = (unsigned int *)calloc(sizeof(unsigned int), NUM_BINS);
  resultRef = (unsigned int *)calloc(sizeof(unsigned int), NUM_BINS);

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)

  for (int i = 0; i < inputLength; i++)
  {
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ Insert code below to create reference result in CPU
  for (int i = 0; i < inputLength; i++)
  {
    if (resultRef[hostInput[i]] < 127)
    {
      resultRef[hostInput[i]]++;
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, sizeof(unsigned int) * inputLength);
  cudaMalloc(&deviceBins, sizeof(unsigned int) * NUM_BINS);

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results ?

  //@@ Initialize the grid and block dimensions here
  dim3 grid1((inputLength + TPB - 1) / TPB);
  dim3 block1(TPB);

  //@@ Launch the GPU Kernel here
  timer.start();
  histogram_kernel<<<grid1, block1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
  timer.stop("kernel");

  //@@ Initialize the second grid and block dimensions here

  dim3 grid2((NUM_BINS + TPB - 1) / TPB);
  dim3 block2(TPB);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<grid2, block2>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference

  FILE *file = fopen("data.txt", "w");

  float meanError = 0;
  for (int i = 0; i < NUM_BINS; i++)
  {
    meanError += fabs(resultRef[i] - hostBins[i]);
    fprintf(file, "%d,", hostBins[i]);
  }
  fclose(file);

  meanError /= NUM_BINS;

  printf("Mean error is %f \n", meanError);

  //@@ Free the GPU memory here
  cudaFree(deviceBins);
  cudaFree(deviceInput);

  //@@ Free the CPU memory here
  free(hostBins);
  free(hostInput);
  free(resultRef);

  return 0;
}
