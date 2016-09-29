#ifndef ROUGHSORT_H
#define ROUGHSORT_H

#include <climits>
#include <cstddef>
#include <cstdint>

extern int32_t (*randInt)();
inline int32_t randIntN(int m) { return randInt() % m; }
void randInit();
int32_t xorsh();
int64_t xorsh64();
int randLen(int min, int max);
void randArray(int32_t* const a, const int n);

void referenceSort(int32_t* const a, const int n);
void hostMergesort(int32_t* const a, int32_t* const b, const int n);
void hostMergesortUp(int32_t* const a, int32_t* const b, const int n);
void hostQuicksort(int32_t* const a, const int n);
void buildLR(const int32_t* const a, int32_t* const b, const int n);
void buildRL(const int32_t* const a, int32_t* const c, const int n);
void buildDM(const int32_t* const b, const int32_t* const c, int32_t* const d,
             const int n);
int rough(const int32_t* const d, const int n);
void hostRoughsort(int32_t* const a, const int n);

void devMergesort(int32_t* const a, const int n);
void devRoughsort(int32_t* const a, const int n);

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

void devSquare(int32_t* const a, const int n);

#endif
