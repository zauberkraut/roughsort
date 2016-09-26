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
      "  -g        Skip the sequential, non-GPU algorithms\n"
      "  -n <len>  Specify random unsorted array length (default: %d)\n"
      "  -t        Confirm sorted array is in order\n",
      DEFAULT_ARRAY_LEN);
  exit(1);
}

bool testArrayEq(const int32_t* const a, const int32_t* const b, const int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

int32_t* g_mergesortBuffer;
} // end anonymous namespace

int main(int argc, char* argv[]) {
  msg("Roughsort Demonstration\nA. Pfaff, J. Treadwell 2016\n");

  bool runHostSorts = true;
  const bool runDevSorts = false; // TODO
  bool testSorted = false;
  int arrayLen = DEFAULT_ARRAY_LEN;

  int option;
  while ((option = getopt(argc, argv, "hgn:t")) != -1) {
    switch (option) {
      char cbuf[21];

    case 'h':
      usage();
      break;
    case 'g':
      runHostSorts = false;
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

  randInit();

  msg("allocating storage...", arrayLen);
  auto unsortedArray   = new int32_t[arrayLen],
       sortingArray    = new int32_t[arrayLen],
       referenceArray  = testSorted ? new int32_t[arrayLen] : nullptr;
  g_mergesortBuffer = new int32_t[arrayLen];

  const auto arraySize = sizeof(*unsortedArray) * arrayLen;
  msg("generating a random array of %d integers...", arrayLen);
  randArray(unsortedArray, arrayLen);

  if (testSorted) {
    msg("sorting reference array for comparison later...");
    memcpy(referenceArray, unsortedArray, arraySize);
    timespec start;
    getTime(&start);
    referenceSort(referenceArray, arrayLen);
    msg("%15s took %d ms", "cstdlib qsort()", msSince(&start));
  }

  auto hostMergesortWrap = [](int32_t* const a, const int n) {
    hostMergesort(a, g_mergesortBuffer, n);
  };

  struct {
    const char* name;
    void (*sort)(int32_t* const, const int);
    bool runTest;
  } benchmarks[] = {
    {"CPU Mergesort",  hostMergesortWrap, runHostSorts},
    {"CPU Quicksort",  hostQuicksort,     runHostSorts},
    {"CPU Roughsort",  hostRoughsort,     false && runHostSorts},
    {"GPU Mergesort",  devMergesort,      runDevSorts},
    {"GPU Roughsort",  devRoughsort,      runDevSorts}
  };
  const int benchmarksLen = sizeof(benchmarks) / sizeof(*benchmarks);

  msg("running sort algorithm benchmarks...");

  for (int i = 0; i < benchmarksLen; i++) {
    if (benchmarks[i].runTest) {
      memcpy(sortingArray, unsortedArray, arraySize);
      timespec start;
      getTime(&start);
      benchmarks[i].sort(sortingArray, arrayLen);
      auto ms = msSince(&start);
      msg("%15s took %d ms%s", benchmarks[i].name, ms,
          testSorted && !testArrayEq(sortingArray, referenceArray, arrayLen) ?
            " BUT IS BROKEN" : "");
    }
  }

  delete[] unsortedArray;
  delete[] sortingArray;
  delete[] referenceArray; // nullptr deletion is safe
  delete[] g_mergesortBuffer;

  return 0;
}
