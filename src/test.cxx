/* test.cxx: unit tests */

#include <csetjmp>
#include <cstdarg>
#include <cstddef>
extern "C" {
#include <cmocka.h>
}
#include <cstring>
#include "roughsort.h"

namespace {

enum {
  MIN_RAND_ARRAY_LEN = 1 << 10,
  MAX_RAND_ARRAY_LEN = 1 << 20
};

/* Random length for an unsorted array. */
int randLen() {
  return randIntN(MAX_RAND_ARRAY_LEN - MIN_RAND_ARRAY_LEN + 1) +
         MIN_RAND_ARRAY_LEN;
}

/* Randomizes an integer array. */
void randArray(int* const a, const int n) {
  for (int i = 0; i < n; i++) {
    a[i] = randInt();
  }
}

int test0[]  = {},
    test1[]  = {0},
    test2[]  = {0, 0},
    test3[]  = {0, 1},
    test4[]  = {1, 0},
    test5[]  = {0, 0, 0},
    test6[]  = {0, 0, 1},
    test7[]  = {0, 1, 0},
    test8[]  = {0, 1, 1},
    test9[]  = {1, 0, 0},
    test10[] = {1, 0, 1},
    test11[] = {1, 1, 0},
    test12[] = {1, 1, 1},
    test13[] = {0, 1, 2},
    test14[] = {0, 2, 1},
    test15[] = {1, 0, 2},
    test16[] = {1, 2, 0},
    test17[] = {2, 1, 0},
    test18[] = {2, 0, 1},
    test19[] = {1, 8, 2, 4, 6, 7, 5, 9, 0, 3},
    test20[] = {1, 8, 8, 4, 2, 7, 5, 9, 0, 7};

class Test {
public:
  int* a;
  int n;
};

#define ALEN(a) (sizeof(a) / sizeof(*a))
#define ATEST(a) {a, ALEN(a)}
Test tests[] = {
  ATEST(test0), ATEST(test1), ATEST(test2), ATEST(test3), ATEST(test4),
  ATEST(test5), ATEST(test6), ATEST(test7), ATEST(test8), ATEST(test9),
  ATEST(test10), ATEST(test11), ATEST(test12), ATEST(test13), ATEST(test14),
  ATEST(test15), ATEST(test16), ATEST(test17), ATEST(test18), ATEST(test19),
  ATEST(test20),
};
const int testCount = ALEN(tests) + 1;

void runTest(void** state, void (*sort)(int*, int)) {
  static int a[MAX_RAND_ARRAY_LEN];

  for (int i = 0; i < testCount; i++) {
    auto randTest = i == testCount - 1;
    auto test = randTest ? Test{0, randLen()} : tests[i];

    if (randTest) {
      randArray(a, test.n);
    } else {
      memcpy(a, test.a, sizeof(*test.a) * test.n);
    }

    sort(a, test.n);

    int prev = 0;
    for (int i = 0; i < test.n; i++) {
      auto x = a[i];
      assert_true(prev <= x);
      prev = x;
    }
  }
}

void testCudaMemory(void** state) {
  const int n = randLen();
  const size_t size = sizeof(int) * n;
  int* devA = (int*)cuMalloc(size);
  cuClear(devA, size);

  auto a = new int[n];
  cuDownload(a, devA, size);
  for (int i = 0; i < n; i++) {
    assert_false(a[i]);
  }

  auto b = new int[n];
  randArray(a, n);
  cuUpload(devA, a, size);
  cuDownload(b, devA, size);
  for (int i = 0; i < n; i++) {
    assert_int_equal(a[i], b[i]);
  }

  cuFree(devA);
  delete[] a;
  delete[] b;
}

void testCudaKernel(void** state) {
  int a[] = {1, 2, 3}, expected[] = {1, 4, 9};
  const int n = sizeof(a) / sizeof(*a);
  int* devA = (int*)cuMalloc(sizeof(a));
  cuUpload(devA, a, sizeof(a));
  cuSquare(devA, n);
  cuDownload(a, devA, sizeof(a));

  for (int i = 0; i < n; i++) {
    assert_int_equal(a[i], expected[i]);
  }

  cuFree(devA);
}

void testHostMergesort(void** state) { runTest(state, hostMergesort); }
void testHostQuicksortC(void** state) { runTest(state, hostQuicksortC); }
void testHostQuicksort(void** state) { runTest(state, hostQuicksort); }
void testHostRoughsort(void** state) {}
void testCudaMergesort(void** state) {}
void testCudaQuicksort(void** state) {}
void testCudaRoughsort(void** state) {}

/* Ensure that this PRNG never emits the same term twice within one period.
   This test might take a few minutes to run. */
void testXorshift(void** state) {
  auto buckets = new unsigned char[1 << 29]; // one bit marker per 32-bit term
  memset(buckets, 0, 1 << 29);

  for (int64_t iter = 0; iter < ((int64_t)1 << 32) - 1; iter++) {
    unsigned x = (unsigned)xorsh();
    unsigned byte = x >> 3;
    unsigned bit = x & 7;
    assert_false(buckets[byte] & 1 << bit);
    buckets[byte] |= 1 << bit;
  }

  delete[] buckets;
}

} // end anonymous namespace

int main() {
  randInit();

  const struct CMUnitTest tests[] = {
    cmocka_unit_test(testCudaMemory),
    cmocka_unit_test(testCudaKernel),
    cmocka_unit_test(testHostMergesort),
    cmocka_unit_test(testHostQuicksortC),
    cmocka_unit_test(testHostQuicksort),
    cmocka_unit_test(testHostRoughsort),
    cmocka_unit_test(testCudaMergesort),
    cmocka_unit_test(testCudaQuicksort),
    cmocka_unit_test(testCudaRoughsort),
    cmocka_unit_test(testXorshift),
  };

  return cmocka_run_group_tests(tests, 0, 0);
}
