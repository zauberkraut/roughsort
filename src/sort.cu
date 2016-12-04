/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include "roughsort.h"

template<typename T>
class forceMergesort : thrust::binary_function<T, T, bool> {
public:
  __device__ bool operator()(const T& a, const T& b) const {
    return a < b;
  }
};

void devMergesort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::sort(devA, devA + n, forceMergesort<int32_t>());
  cudaDeviceSynchronize();
}

void devRadixsort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::sort(devA, devA + n);
  cudaDeviceSynchronize();
}

static __global__ void kernBatchSort(int32_t* const a, const int n,
                                     const int offset, const int len) {
  const int start = min(n, offset +
                           len*(blockIdx.x*blockDim.x + threadIdx.x));
  const int end   = min(n, start + len);

  thrust::sort(thrust::seq, a + start, a + end);
}

/* Parallel roughsort implementation. */
void devRoughsort(int32_t* const a, const int radius, const int n) {
  if (radius == 0) {
    return;
  }
  const int blockDim = 512;

  cudaError_t r;
  int k = radius, p = 0;
  do {
    const int len = k << 1;
    const int nSegments = (n + len - 1)/len;
    const int nBlocks = (nSegments + blockDim-1)/blockDim;
    kernBatchSort<<<nBlocks, blockDim>>>(a, n, 0, len);
    kernBatchSort<<<nBlocks, blockDim>>>(a, n, k, len);
    r = cudaGetLastError();

    k = radius / (2 << p++);
  } while (k > 1 && r == cudaSuccess);

  if (r != cudaSuccess) {
    printf("KERNEL ERROR: %s\n", cudaGetErrorString(r));
  }
  cudaDeviceSynchronize();
}
