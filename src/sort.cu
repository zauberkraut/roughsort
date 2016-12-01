/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include "roughsort.h"

const int NSTREAMS = 64;

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

/*
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

static int devRadius(const int32_t* const a, const int n) {
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
*/

/* Produces a (k - 1)/2-sorted sequence from a k-sorted one. */
static void devHalve(thrust::device_ptr<int32_t> a, const int radius,
                     const int n, cudaStream_t* streams) {
  thrust::device_ptr<int32_t> end = a + n;
  const int tailLen = n%radius;
  thrust::device_ptr<int32_t> endSeg = end - tailLen - radius;

  thrust::device_ptr<int32_t> offset;
  int seg;

  for (offset = a, seg = 0; offset <= endSeg;
       offset += radius, seg = (seg + 1)%NSTREAMS) {
    thrust::sort(thrust::cuda::par.on(streams[seg]), offset, offset + radius);
  }
  if (tailLen) {
    thrust::sort(thrust::cuda::par.on(streams[seg]), offset, end);
  }

  cudaDeviceSynchronize();
  for (offset = a + (radius>>1), seg = 0; offset <= endSeg;
       offset += radius, seg = (seg + 1)%NSTREAMS) {
    thrust::sort(thrust::cuda::par.on(streams[seg]), offset, offset + radius);
  }
  if (tailLen) {
    thrust::sort(thrust::cuda::par.on(streams[seg]), offset, end);
  }

  cudaDeviceSynchronize();
  for (offset = a, seg = 0; offset <= endSeg;
       offset += radius, seg = (seg + 1)%NSTREAMS) {
    thrust::sort(thrust::cuda::par.on(streams[seg]), offset, offset + radius);
  }
  if (tailLen) {
    thrust::sort(thrust::cuda::par.on(streams[seg]), offset, end);
  }
}

/* Parallel roughsort implementation.
   ASSUMES k > 1 */
void devRoughsort(int32_t* const a, const int radius, const int n) {
  thrust::device_ptr<int32_t> devA(a);
  cudaStream_t* streams = new cudaStream_t[NSTREAMS];

  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamCreate(streams + i);
  }

  if (!radius || n < 2) {
    return;
  }

  int k = radius, p = 0;
  do {
    devHalve(devA, k, n, streams);
    k = radius / (2 << p++);
  } while (k > 1);

  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  delete[] streams;
}
