/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include "thrust/device_ptr.h"
#include "thrust/sort.h"
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
  // TODO
}
