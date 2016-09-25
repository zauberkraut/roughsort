#ifndef ROUGHSORT_H
#define ROUGHSORT_H

#include <cstddef>
#include <cstdint>

extern int (*randInt)();
inline int randIntN(int m) { return randInt() % m; }
void randInit();
int xorsh();
int xorsh64();
void randArray(int* const a, const int n);

void hostMergesort(int* const a, const int n);
void hostQuicksort(int* const a, const int n);
void hostQuicksortC(int* const a, const int n);
void hostRoughSort(int* const a, const int n);

void cudaMergesort(int* const a, const int n);
void cudaQuicksort(int* const a, const int n);
void cudaRoughsort(int* const a, const int n);

void msg(const char* s, ...);
void warn(const char* s, ...);
[[ noreturn ]] void fatal(const char* s, ...);
double mibibytes(size_t size);

size_t cuMemAvail();
void* cuMalloc(size_t size);
void cuFree(void* p);
void cuClear(void* p, size_t size);
void cuUpload(void* devDst, const void* hostSrc, size_t size);
void cuDownload(void* hostDst, const void* devSrc, size_t size);
void cuPin(void* p, size_t size);
void cuUnpin(void* p);

void cuSquare(int* a, int n);

#endif
