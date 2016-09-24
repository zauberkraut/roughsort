/* random.cxx: k-sorted random sequence generator */

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
   Requires an Ivy Bridge or newer x86 CPU. Requires no seeding. */
int rdRand() {
  unsigned r;
  if (_rdrand32_step(&r)) {
    return (int)(r >> 1);
  }
  // I'm not sure how often this should happen...
  warn("RDRAND ran out of entropy; sourcing from rand()");
  return (int)r ^ rand();
}

[[ noreturn ]] int nullRand() {
  fatal("randInt() was invoked before initialization!");
  exit(-1);
}

}

/* Either high-end RDRAND or lousy rand(). */
int (*randInt)() = nullRand;

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
   repeating. */
int xorsh() {
	auto x = state;
	x ^= x << 5;
	x ^= x >> 17;
	x ^= x << 13;
	state = x;
	return (int)x;
}

/* 64-bit version of the above, just in case. */
int xorsh64() {
	auto x = state64;
	x ^= x << 12;
	x ^= x >> 7;
	x ^= x << 13;
	state64 = x;
	return (int64_t)x;
}
