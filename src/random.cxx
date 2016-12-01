/* random.cxx: (P)RNGs and random sequence generator. */

#include <algorithm>
#include <chrono>
#include <cpuid.h>
#include <immintrin.h>
#include <random>
#include <thread>
#include "roughsort.h"

namespace {

/* Tests for CPU support of the RDRAND instruction. */
bool rdRandSupported() {
  unsigned eax, ebx, ecx, edx;
  return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ecx & bit_RDRND;
}

/* Uses RDRAND instruction to generate high-quality random integers.
   Intended for use in the creation of k-sorted sequences for a given k.
   Requires an Ivy Bridge or newer x86 CPU. Requires no seeding. */
int32_t rdRand() {
  static thread_local uint32_t w[2];
  static thread_local int n = 0;
  typedef long long unsigned int* p;

  if (!n) {
    _rdrand64_step(reinterpret_cast<p>(w)); // assume entropy suffices
    n = 2;
  }
  return (int32_t)(w[--n] >> 1);
}

int32_t mtRand() {
  static thread_local std::mt19937 // thread safe
    rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return (int32_t)(rng() >> 1);
}

[[ noreturn ]] int32_t nullRand() {
  fatal("randInt() was invoked before initialization!");
}

/* Either high-end RDRAND or mt19937. */
int32_t (*randInt)() = nullRand;

void kPerturb(int32_t* const a, const int k, const int n) {
  int nDisplaced = 0;

  while (!nDisplaced) {
    for (int i = k; i < n; i++) {
      if (randInt() & 1) {
        auto x = a[i];
        a[i] = a[i - k];
        a[i - k] = x;

        nDisplaced++;
        i += k;
      }
    }
  }
}

class ShuffleRNG {
public:
  typedef unsigned result_type;
  static unsigned min() { return 0; }
  static unsigned max() { return INT32_MAX; }
  unsigned operator()() {
    return (unsigned)randInt();
  }
};

void kShuffle(int32_t* const a, const int k, const int n) {
  const int l = k + 1;
  const int r = randInt()%(n - k);
  const bool pushRight = randInt() & 1;
  int32_t* const end = a + n;
  int32_t* const p = pushRight ? a + r : end - 1 - r - k;
  int32_t* const q = p + k;

  // perform guaranteed k-displacement
  const int32_t tmp = *p;
  *p = *q;
  *q = tmp;

  for (int32_t* offset = a; offset < p; offset += l) {
    int32_t* const cap = std::min(p, offset + l);
    std::shuffle(offset, cap, ShuffleRNG());
  }

  std::shuffle(p + 1, q, ShuffleRNG());

  for (int32_t* offset = q + 1; offset < end; offset += l) {
    int32_t* const cap = std::min(end, offset + l);
    std::shuffle(offset, cap, ShuffleRNG());
  }
}

} // end anonymous namespace

/* Selects general-purpose RNG and seeds it if necessary. */
void randInit(bool forceMT) {
  if (rdRandSupported() && !forceMT) {
    randInt = rdRand;
  } else { // fall back on the C++ MT generator and seed it
    warn("Your CPU is old and an embarrassment; falling back on mt19937");
    randInt = mtRand;
  }
}

/* Random length for an unsorted array. This might not be uniform. */
int randLen(int min, int max) { return randInt()%(max - min + 1) + min; }

const int NTHREADS = 8;
/* Memory-coalesced random array filler. */
void randRun(int thId, int32_t* a, int n) {
  for (int i = thId; i < n; i += NTHREADS) {
    a[i] = randInt();
  }
}

/* Randomizes an integer array with distinct 32-bit elements. Assumes k < n. */
void randArray(int32_t* const a, const int k, const int n, bool shuffle) {
  std::thread runs[NTHREADS];

  for (int i = 0; i < NTHREADS; i++) {
    runs[i] = std::thread(randRun, i, a, n);
  }

  for (int i = 0; i < NTHREADS; i++) {
    runs[i].join();
  }

  if (k >= 0) {
    hostQuicksort(a, n);

    if (k > 0) {
      shuffle ? kShuffle(a, k, n) : kPerturb(a, k, n);
    }
  }

  const int radius = hostRadius(a, n);
  printf("random array is %d-sorted\n", radius);
  if (radius != k && k > -1) {
    fatal("...but we needed it %d-sorted!", k);
  }
}
