/* util.cu: CUDA convenience functions. */

#include "roughsort.h"

// TODO: propagate __FILE__ and __LINE__ from caller if GDB proves inadequate
#define CHECK(r) cuCheck(r, __FILE__, __LINE__)

inline void cuCheck(cudaError_t r, const char* fname, const size_t lnum) {
  if (r != cudaSuccess) {
    fatal("CUDA error at line %d in %s\n", lnum, fname);
  }
}

size_t cuMemAvail() {
  size_t free, total;
  CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

void* cuMalloc(size_t size) {
  void* p;
  CHECK(cudaMalloc(&p, size));
  return p;
}

void cuFree(void* p) {
  CHECK(cudaFree(p));
}

void cuClear(void* p, size_t size) {
  CHECK(cudaMemset(p, 0, size));
}

void cuUpload(void* devDst, const void* hostSrc, size_t size) {
  CHECK(cudaMemcpy(devDst, hostSrc, size, cudaMemcpyHostToDevice));
}

void cuDownload(void* hostDst, const void* devSrc, size_t size) {
  CHECK(cudaMemcpy(hostDst, devSrc, size, cudaMemcpyDeviceToHost));
}

void cuPin(void* p, size_t size) {
  CHECK(cudaHostRegister(p, size, cudaHostRegisterPortable));
}

void cuUnpin(void* p) {
  CHECK(cudaHostUnregister(p));
}
