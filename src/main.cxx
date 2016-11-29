/* main.cxx */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include "roughsort.h"

// for dealing with getopt arguments
extern char *optarg;
extern int optind;

namespace {

enum {
  MIN_ARRAY_LEN = 2,
  MAX_ARRAY_LEN = (1 << 30) + (1 << 29) + (1 << 28), // ~1.75 bil. ints, 7 GiB
  MIN_RAND_LEN = 1 << 10,
  MAX_RAND_LEN = 1 << 24
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
      "  -k <int>  Set the k-sortedness of the random array (default: don't k-sort)\n"
      "  -m        Force usage of software Mersenne Twister RNG\n"
      "  -n <len>  Set randomized array length (default: random)\n"
      "  -s        Shuffle each (k + 1)-segment\n"
      "  -t        Confirm sorted array is in order\n");
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

} // end anonymous namespace

int main(int argc, char* argv[]) {
  msg("Roughsort Demonstration\nA. Pfaff, J. Treadwell 2016\n");

  bool runHostSorts = true;
  int k = -1;
  bool forceMT = false;
  int arrayLen = 0;
  bool shuffle = false;
  bool testSorted = false;

  int option;
  while ((option = getopt(argc, argv, "hgk:mn:st")) != -1) {
    switch (option) {
    case 'h':
      usage();

    case 'g':
      runHostSorts = false;
      break;

    case 'k':
      k = (int)parseInt(10, 0, MAX_ARRAY_LEN, "invalid k-sortedness");
      break;

    case 'm':
      msg("forcing usage of software RNG");
      forceMT = true;
      break;

    case 'n':
      arrayLen = (int)parseInt(10, MIN_ARRAY_LEN, MAX_ARRAY_LEN,
                               "invalid unsorted array length");
      break;

    case 's':
      msg("random array shuffling enabled");
      shuffle = true;
      break;

    case 't':
      testSorted = true;
      break;

    case '?': // deal with ill-formed parameterless option
    default:  // ...or invalid option
      usage();
    }
  }
  if (optind < argc) { // reject extra arguments
    fatal("unexpected argument: %s", argv[optind]);
  }
  if (k >= arrayLen) {
    fatal("an array of length %d can't be %d-sorted!", arrayLen, k);
  }

  randInit(forceMT);
  if (!arrayLen) {
    arrayLen = randLen(MIN_RAND_LEN, MAX_RAND_LEN);
  }

  const auto arraySize = 4*arrayLen;
  msg("allocating storage...");
  int32_t* const hostUnsortedArray = new int32_t[arrayLen];
  int32_t* const hostSortingArray  = new int32_t[arrayLen];
  int32_t* hostReferenceArray = nullptr;
  int32_t* const devSortingArray = (int32_t*)cuMalloc(arraySize);

  msg("generating a random array of %d integers...", arrayLen);
  randArray(hostUnsortedArray, k, arrayLen, shuffle);

  struct {
    const char* name;
    void (*sort)(int32_t* const, const int);
    bool runTest;
    bool onGPU;
  } benchmarks[] = {
    {"CPU Mergesort", hostMergesort, runHostSorts, false},
    {"CPU Quicksort", hostQuicksort, runHostSorts, false},
    {"CPU Roughsort", hostRoughsort, runHostSorts, false},
    {"GPU Mergesort", devMergesort,  true, true},
    {"GPU Quicksort", devQuicksort,  true, true},
    {"GPU Roughsort", devRoughsort,  false, true}
  };

  msg("running sort algorithm benchmarks...");

  for (auto const bench : benchmarks) {
    if (bench.runTest) {
      const bool referenceSort = testSorted && hostReferenceArray == nullptr;

      if (bench.onGPU) {
        cuUpload(devSortingArray, hostUnsortedArray, arraySize);
      } else {
        memcpy(hostSortingArray, hostUnsortedArray, arraySize);
      }

      timespec start;
      getTime(&start);
      bench.sort(bench.onGPU ? devSortingArray : hostSortingArray, arrayLen);
      auto ms = msSince(&start);

      auto resultMsg = "";

      if (referenceSort) {
        /* NOTE: We could maintain separate reference and unsorted arrays on the
                 GPU, but we probably can't afford the extra device RAM and
                 won't need the speedup. */
        hostReferenceArray = new int32_t[arrayLen];
        msg("copying sorted array to reference for testing further results...");
        if (bench.onGPU) {
          cuDownload(hostReferenceArray, devSortingArray, arraySize);
        } else {
          memcpy(hostReferenceArray, hostSortingArray, arraySize);
        }
      } else if (testSorted) {
        if (bench.onGPU) {
          cuDownload(hostSortingArray, devSortingArray, arraySize);
        }
        if (!testArrayEq(hostSortingArray, hostReferenceArray, arrayLen)) {
          resultMsg = " BUT IS BROKEN";
        }
      }

      msg("  %s took %d ms%s", bench.name, ms, resultMsg);
    }
  }

  delete[] hostUnsortedArray;
  delete[] hostSortingArray;
  delete[] hostReferenceArray; // nullptr deletion is safe
  cuFree(devSortingArray);

  return 0;
}
