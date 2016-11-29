/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "roughsort.h"

void devMergesort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::stable_sort(devA, devA + n);
}

void devQuicksort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::sort(devA, devA + n);
}

void devRoughsort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
}

static __global__ void kernRough(const int32_t* const a, const int n,
                                 int32_t* const b, int32_t* const c,
                                 int* const d) {
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id >= n) {
    return;
  }
  const int logn = 31 - __clz(n);

  int32_t cur = b[id] = c[id] = a[id];
  // max prefix
  int32_t end = b[n - 1];
  for (int r = 0; cur > end && r <= logn; r++) {
    const int j = id - (1 << r);
    if (j < 0) {
      break;
    }
    cur = b[id] = max(cur, b[j]);
  }

  __syncthreads();

  // min prefix
  end = c[n - 1];
  for (int r = 0; cur > end && r <= logn; r++) {
    const int j = id + (1 << r);
    if (j >= n) {
      break;
    }
    cur = c[id] = min(cur, c[j]);
  }

  __syncthreads();

  int dist = 0;
  for (int j = id; j >= 0; j--) { // cur = c[id]
    if (cur <= b[j] && (j == 0 || cur >= b[j - 1])) {
      dist = id - j;
    }
  }

  __syncthreads();
  d[id] = dist;
}

int devRough(const int32_t* const a, const int n) {
  const int nThreadsPerBlock = 1024;
  const int nBlocks = (n + nThreadsPerBlock - 1)/nThreadsPerBlock;

  int32_t* b = (int32_t*)cuMalloc(4*n);
  int32_t* c = (int32_t*)cuMalloc(4*n);
  int* d = (int*)cuMalloc(sizeof(int)*n);

  kernRough<<<nBlocks, nThreadsPerBlock>>>(a, n, b, c, d);

  cudaDeviceSynchronize();
  thrust::device_ptr<int32_t> thrustD(d);
  const int k = *thrust::max_element(thrustD, thrustD + n);

  cuFree(b);
  cuFree(c);
  cuFree(d);

  return k;
}
