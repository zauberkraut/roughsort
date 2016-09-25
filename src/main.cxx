/* main.cxx */

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include "roughsort.h"

// for dealing with getopt arguments
extern char *optarg;
extern int optopt;

namespace {

enum {
  MIN_ARRAY_LEN = 2,
  MAX_ARRAY_LEN = (1 << 30) + (1 << 29) + (1 << 28), // ~1.75 bil. ints, 7 GiB
  DEFAULT_ARRAY_LEN = 1 << 24 // ~16 mil. ints, 64 MiB
};

/* Parses an integer argument of the given radix from the command line, aborting
   after printing errMsg if an error occurs or the integer exceeds the given
   bounds. */
int parseInt(int radix, int min, int max, const char* errMsg) {
  char* parsePtr = nullptr;
  int i = strtol(optarg, &parsePtr, radix);
  if ((size_t)(parsePtr - optarg) != strlen(optarg) || i < min || i > max ||
      ERANGE == errno) {
    fatal(errMsg);
  }
  return i;
}

void getTime(timespec* start) {
  clock_gettime(CLOCK_MONOTONIC, start);
}

int msSince(timespec* start) {
  timespec time;
  getTime(&time);
  return (time.tv_sec - start->tv_sec)*1000 +
         (time.tv_nsec - start->tv_nsec)/1.e6;
}

[[ noreturn ]] void usage() {
  msg("roughsort [options]\n"
      "  -h        These instructions\n"
      "  -c        Perform all sorting on host without the GPU\n"
      "  -n <len>  Specify random unsorted array length (default: %d)\n"
      "  -t        Confirm sorted array is in order\n",
      DEFAULT_ARRAY_LEN);
  exit(1);
}

bool testArrayEq(const int* const a, const int* const b, const int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

int* g_mergesortBuffer;
} // end anonymous namespace

int main(int argc, char* argv[]) {
  msg("Roughsort Demonstration\nA. Pfaff, J. Treadwell 2016\n");

  bool useGPU = true;
  bool testSorted = false;
  int arrayLen = DEFAULT_ARRAY_LEN;

  int option;
  while ((option = getopt(argc, argv, "hcn:t")) != -1) {
    switch (option) {
      char cbuf[21];

    case 'h':
      usage();
      break;
    case 'c':
      useGPU = false;
      break;
    case 't':
      testSorted = true;
      break;
    case 'n':
      arrayLen = (int)parseInt(10, MIN_ARRAY_LEN, MAX_ARRAY_LEN,
                              "invalid unsorted array length");
      break;

    case '?': // deal with ill-formed parameterless option
      switch (optopt) {
      case 'n':
        fatal("option -%c missing argument", optopt);
      }
      // fallthrough
    default: // report invalid option character
      if (isprint(optopt)) {
        snprintf(cbuf, sizeof(cbuf), "%c", optopt);
      } else {
        snprintf(cbuf, sizeof(cbuf), "<0x%x>", optopt);
      }
      fatal("invalid option: -%s", cbuf);
    }
  }

  if (useGPU) {
    fatal("work in progress");
  }

  randInit();

  msg("allocating storage...", arrayLen);
  auto unsortedArray   = new int[arrayLen],
       sortingArray    = new int[arrayLen],
       referenceArray  = testSorted ? new int[arrayLen] : nullptr;
  g_mergesortBuffer = new int[arrayLen];
  const auto arraySize = sizeof(*unsortedArray) * arrayLen;
  msg("generating a random array of %d integers...", arrayLen);
  randArray(unsortedArray, arrayLen);

  if (referenceArray) {
    msg("sorting reference array for comparison later...");
    memcpy(referenceArray, unsortedArray, arraySize);
    hostQuicksortC(referenceArray, arrayLen);
  }

  auto hostMergesortWrap = [](int* const a, const int n) {
    hostMergesort(a, g_mergesortBuffer, n);
  };

  struct {
    const char* name;
    void (*sort)(int* const, const int);
  } benchmarks[] = {
    {"CPU  Mergesort",           hostMergesortWrap},
    {"CPU  Quicksort (cstdlib)", hostQuicksortC},
    {"CPU  Quicksort",           hostQuicksort},
  /*{"CPU  Roughsort",           hostRoughsort},
    {"CUDA Mergesort",           cudaMergesort},
    {"CUDA Quicksort",           cudaQuicksort},
    {"CUDA Roughsort",           cudaRoughsort},*/
  };
  const int benchmarksLen = sizeof(benchmarks) / sizeof(*benchmarks);

  for (int i = 0; i < benchmarksLen; i++) {
    memcpy(sortingArray, unsortedArray, arraySize);
    timespec start;
    getTime(&start);
    benchmarks[i].sort(sortingArray, arrayLen);
    auto ms = msSince(&start);
    msg("%25s took %d ms%s", benchmarks[i].name, ms,
        testSorted && !testArrayEq(sortingArray, referenceArray, arrayLen) ?
          " BUT IS BROKEN" : "");
  }

  delete[] unsortedArray;
  delete[] sortingArray;
  delete[] referenceArray; // nullptr deletion is safe
  delete[] g_mergesortBuffer;

  return 0;
}
