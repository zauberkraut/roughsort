#ifndef ROUGHSORT_H
#define ROUGHSORT_H

#include <cstdint>

void randInit(bool forceMT);
int randLen(int min, int max);
void randArray(int32_t* const a, const int k, const int n, bool shuffle);

void hostMergesort(int32_t* const a, const int n);
void hostQuicksort(int32_t* const a, const int n);
int hostRough(const int32_t* const d, const int n);
void hostRoughsort(int32_t* const a, const int n);

void devMergesort(int32_t* const a, const int n);
void devQuicksort(int32_t* const a, const int n);
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

#endif
