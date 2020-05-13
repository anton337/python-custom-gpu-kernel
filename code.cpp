#include "code.h"
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

using namespace std;

vector<float> sum(vector<vector<float> > input) {

  int N = input[0].size();
  size_t size = N*sizeof(float);

  CUdevice cuDevice;
  CUcontext cuContext;
  CUmodule cuModule;
  CUfunction sum_kernel;

  float * h_A, * h_B, * h_C;
  CUdeviceptr d_A, d_B, d_C;

  checkCudaErrors(cuInit(0));

  char name[100];
  int devID = gpuGetMaxGflopsDeviceIdDRV();
  checkCudaErrors(cuDeviceGet(&cuDevice, devID));
  cuDeviceGetName(name, 100, cuDevice);
  printf("> Using CUDA Device [%d]: %s\n", devID, name);

  cuDeviceGet(&cuDevice, devID);

  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

  checkCudaErrors(cuModuleLoad(&cuModule, "sum_kernel.ptx"));

  checkCudaErrors(cuModuleGetFunction(&sum_kernel, cuModule, "sum"));

  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  memcpy(h_A, &(input[0])[0], size);
  memcpy(h_B, &(input[1])[0], size);

  checkCudaErrors(cuMemAlloc(&d_A, size));
  checkCudaErrors(cuMemAlloc(&d_B, size));
  checkCudaErrors(cuMemAlloc(&d_C, size));

  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
  checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
  void * args[] = {&d_A, &d_B, &d_C, &N};

  checkCudaErrors(cuLaunchKernel( sum_kernel, blocksPerGrid, 1, 1
                                , threadsPerBlock, 1, 1, 0, NULL
                                , args, NULL ));

  checkCudaErrors(cuCtxSynchronize());

  checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

  vector<float> sums;
  sums.resize(N);
  memcpy(&sums[0], h_C, size);

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuMemFree(d_C));

  if(h_A) free(h_A);
  if(h_B) free(h_B);
  if(h_C) free(h_C);

  return sums;

}

























