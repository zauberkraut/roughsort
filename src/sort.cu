/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

#include "roughsort.h"

/* Useless and only meant for testing CUDA and GPU operation. */
__global__ void kernSquare(int32_t* const a, const int n) {
  for (int i = 0; i < n; i++) {
    a[i] *= a[i];
  }
}

void devSquare(int32_t* const a, const int n) {
  kernSquare<<<1, 1>>>(a, n);
}

void devMergesort(int32_t* const a, const int n) {
  // TODO
}

void devRoughsort(int32_t* const a, const int n) {
  // TODO
}
