/* main.cxx */

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "roughsort.h"

// for dealing with getopt arguments
extern char *optarg;
extern int optind, opterr, optopt;

namespace {

enum {
  MIN_ARRAY_LEN = 2,
  MAX_ARRAY_LEN = (1 << 30) + (1 << 29) + (1 << 28), // 7 GiB
  DEFAULT_ARRAY_LEN = 1 << 28
};

/* Parses an integer argument of the given radix from the command line, aborting
   after printing errMsg if an error occurs or the integer exceeds the given
   bounds. */
int parseInt(int radix, int min, int max, const char* errMsg) {
  char* parsePtr = 0;
  int i = strtol(optarg, &parsePtr, radix);
  if ((size_t)(parsePtr - optarg) != strlen(optarg) || i < min || i > max ||
      ERANGE == errno) {
    fatal(errMsg);
  }
  return i;
}

[[ noreturn ]] void usage() {
  msg("roughsort [options]\n"
      "  -h        These instructions\n"
      "  -c        Perform all sorting on host without the GPU\n"
      "  -n <len>  Specify random unsorted array length (default: %d)\n",
      DEFAULT_ARRAY_LEN);
  exit(1);
}

} // end anonymous namespace

int main(int argc, char* argv[]) {
  msg("Roughsort Demonstration\nA. Pfaff, J. Treadwell 2016\n");

  bool useGPU = true;
  int arrayLen = DEFAULT_ARRAY_LEN;

  int option;
  while ((option = getopt(argc, argv, "hcn:")) != -1) {
    switch (option) {
      char cbuf[21];

    case 'h':
      usage();
      break;
    case 'c':
      useGPU = false;
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

  msg("generating a random array of %d integers...", arrayLen);

  return 0;
}
