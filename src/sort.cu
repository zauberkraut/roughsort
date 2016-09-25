/* sort.cu: CUDA kernels for parallel sorts on an Nvidia GPU. */

/* Useless and only meant for testing CUDA and GPU operation. */
__global__ void kernSquare(int* a, const int n) {
  for (int i = 0; i < n; i++) {
    a[i] *= a[i];
  }
}

void cuSquare(int* a, int n) {
  kernSquare<<<1, 1>>>(a, n);
}
