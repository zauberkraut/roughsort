/* random.cxx: (P)RNGs and random sequence generator. */

#include <climits>
#include <cpuid.h>
#include <ctime>
#include <immintrin.h>
#include <cstdlib>
#include "roughsort.h"

namespace {

unsigned state;
uint64_t state64;

/* Tests for CPU support of the RDRAND instruction. */
bool rdRandSupported() {
  unsigned eax, ebx, ecx, edx;
  return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ecx & bit_RDRND;
}

/* Uses RDRAND instruction to generate high-quality random integers.
   Intended for use in the creation of k-sorted sequences for a given k.
   Requires an Ivy Bridge or newer x86 CPU. Requires no seeding. */
int32_t rdRand() {
  unsigned r;
  if (_rdrand32_step(&r)) {
    // TODO: perhaps save the discarded bit in case we run out of entropy
    return (int32_t)(r >> 1);
  }
  // I'm not sure how often this should happen...
  warn("RDRAND ran out of entropy; sourcing from rand()");
  return (int32_t)(r ^ rand());
}

[[ noreturn ]] int32_t nullRand() {
  fatal("randInt() was invoked before initialization!");
}

} // end anonymous namespace

/* Either high-end RDRAND or lousy rand(). */
int32_t (*randInt)() = nullRand;

/* Selects general-purpose RNG and seeds xorshift (and rand() if there's no
   RDRAND support). */
void randInit() {
  if (rdRandSupported()) {
    randInt = rdRand;
    state = rdRand();
    state64 = rdRand();
  } else { // fall back on lousy cstdlib::rand() and time seeding
    warn("Your CPU is old and an embarrassment; falling back on rand()");
    randInt = rand;
    state = time(0);
    state64 = (((uint64_t)state << 32) + 1) | state;
    srand(state);
  }
  xorsh();   // discard the first term since it's correlated with the seed
  xorsh64();
}

/* High-end and high-speed PRNG over a group structure.
   This PRNG shall generate every nonzero 32-bit integer exactly once before
   repeating and is meant to produce sequences without repeated elements. */
int32_t xorsh() {
  auto x = state;
  x ^= x << 5;
  x ^= x >> 17;
  x ^= x << 13;
  state = x;
  return (int32_t)x;
}

/* 64-bit version of the above, just in case. */
int64_t xorsh64() {
  auto x = state64;
  x ^= x << 12;
  x ^= x >> 7;
  x ^= x << 13;
  state64 = x;
  return (int64_t)x;
}

/* Random length for an unsorted array. */
int randLen(int min, int max) { return randIntN(max - min + 1) + min; }

/* Randomizes an integer array. */
void randArray(int32_t* const a, const int n) {
  for (int i = 0; i < n; i++) {
    a[i] = xorsh();
  }
}
