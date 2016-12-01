/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
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
}

void devRadixsort(int32_t* const a, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  thrust::sort(devA, devA + n);
}

static __device__ bool isSorted;

static __global__ void kernRadius(const int32_t* const a, const int n,
                                  int32_t* const b, int32_t* const c,
                                  int* const d) {
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id >= n) {
    return;
  }
  const int logn = 31 - __clz(n);

  b[id] = c[id] = a[id];

  // max prefix
  for (int r = 0; r <= logn; r++) {
    isSorted = true;
    if (b[id] > b[n - 1]) {
      isSorted = false;
    }
    if (isSorted) {
      break;
    }

    const int j = id - (1 << r);
    if (j >= 0) {
      b[id] = max(b[id], b[j]);
    }
  }

  __syncthreads();

  // min prefix
  for (int r = 0; r <= logn; r++) {
    isSorted = true;
    if (c[id] > c[n - 1]) {
      isSorted = false;
    }
    if (isSorted) {
      break;
    }

    const int j = id + (1 << r);
    if (j < n) {
      c[id] = min(c[id], c[j]);
    }
  }

  __syncthreads();

  int dist = 0;
  for (int j = id - 1; j >= 0; j--) {
    if (c[id] <= b[j] && (j == 0 || c[id] >= b[j - 1])) {
      dist = id - j;
      break;
    }
  }

  __syncthreads();
  d[id] = dist;
}

int devRadius(const int32_t* const a, const int n) {
  const int nThreadsPerBlock = 1024;
  const int nBlocks = (n + nThreadsPerBlock - 1)/nThreadsPerBlock;

  int32_t* b = (int32_t*)cuMalloc(4*n);
  int32_t* c = (int32_t*)cuMalloc(4*n);
  int* d = (int*)cuMalloc(sizeof(int)*n);

  kernRadius<<<nBlocks, nThreadsPerBlock>>>(a, n, b, c, d);

  thrust::device_ptr<int> thrustD(d);
  const int k = *thrust::max_element(thrustD, thrustD + n);

  cuFree(b);
  cuFree(c);
  cuFree(d);

  return k;
}

void devRoughsort(int32_t* const a, const int n) {
  msg("CUDA radius is %d", devRadius(a, n));
}
