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

static __global__ void kernHalve(int32_t* const a, const int radius,
                                 const int n) {
  const int i = min(n, radius*(blockIdx.x*blockDim.x + threadIdx.x));
  const int j = min(n, i + radius);
  const int k = min(n, i + (radius>>1));
  const int l = min(n, k + radius);

  thrust::sort(thrust::seq, a + i, a + j);
  thrust::sort(thrust::seq, a + k, a + l);
  thrust::sort(thrust::seq, a + i, a + j);
}

/* Parallel roughsort implementation.
   ASSUMES k > 1 */
void devRoughsort(int32_t* const a, const int radius, const int n) {
  if (!radius || n < 2) {
    return;
  }

  cudaError_t r;
  int k = radius, p = 0;
  do {
    const int nSegments = (n + k - 1)/k;
    const int nBlocks = (nSegments + 511)/512;
    kernHalve<<<nBlocks, 512>>>(a, k, n);
    r = cudaGetLastError();

    k = radius / (2 << p++);
  } while (k > 1 && r == cudaSuccess);

  if (r != cudaSuccess) {
    printf("KERNEL ERROR: %s\n", cudaGetErrorString(r));
  }
  cudaDeviceSynchronize();
}
