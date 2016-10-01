/* test.cxx: Unit tests. */

#include <climits>
#include <csetjmp>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
extern "C" {
#include <cmocka.h>
}
#include <cstring>
#include "roughsort.h"

namespace {

enum { MAX_TEST_LEN = 64 }; // UPDATE IF A LONGER TEST IS ADDED
int32_t g_test0[]  = {},
        g_test1[]  = {0},
        g_test2[]  = {0, 0},
        g_test3[]  = {0, 1},
        g_test4[]  = {1, 0},
        g_test5[]  = {0, 0, 0},
        g_test6[]  = {0, 0, 1},
        g_test7[]  = {0, 1, 0},
        g_test8[]  = {0, 1, 1},
        g_test9[]  = {1, 0, 0},
        g_test10[] = {1, 0, 1},
        g_test11[] = {1, 1, 0},
        g_test12[] = {1, 1, 1},
        g_test13[] = {0, 1, 2},
        g_test14[] = {0, 2, 1},
        g_test15[] = {1, 0, 2},
        g_test16[] = {1, 2, 0},
        g_test17[] = {2, 1, 0},
        g_test18[] = {2, 0, 1},
        g_test19[] = {1, 8, 2, 4, 6, 7, 5, 9, 0, 3},
        g_test20[] = {1, 8, 8, 4, 2, 7, 5, 9, 0, 7},
        g_testQM0[] = {2, 3, 5, 1, 4, 2, 6, 8, 7, 9, 8, 11, 6, 13, 12,
                       16, 15, 17, 18, 20, 18, 19, 21, 19, 30, 31, 32, 33, 34},
        g_testLR0[] = {2, 3, 5, 5, 5, 5, 6, 8, 8, 9, 9, 11, 11, 13, 13,
                       16, 16, 17, 18, 20, 20, 20, 21, 21, 30, 31, 32, 33, 34},
        g_testRL0[] = {1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6, 6, 12, 12, 15,
                       15, 17, 18, 18, 18, 19, 19, 19, 30, 31, 32, 33, 34},
        g_testDM0[] = {0, 1, 2, 3, 3, 4, 0, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1,
                       0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0},
        g_testQM1[] = {2, 3, 5, 1, 4, 2, 6, 8, 7, 9, 8, 11, 6, 13, 12,
                       16, 15, 17, 18, 20, 18, 19, 21, 19},
        g_testLR1[] = {2, 3, 5, 5, 5, 5, 6, 8, 8, 9, 9, 11, 11, 13, 13,
                       16, 16, 17, 18, 20, 20, 20, 21, 21},
        g_testRL1[] = {1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6, 6, 12, 12, 15,
                       15, 17, 18, 18, 18, 19, 19, 19},
        g_testDM1[] = {0, 1, 2, 3, 3, 4, 0, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1,
                       0, 0, 0, 1, 2, 3, 4};
const int g_testRough = 5;

#define ALEN(a) (sizeof(a) / sizeof(*a))
#define ATEST(a) {a, ALEN(a)}
struct Test { int32_t* const a; const int n; } g_tests[] = {
  ATEST(g_test0), ATEST(g_test1), ATEST(g_test2), ATEST(g_test3),
  ATEST(g_test4), ATEST(g_test5), ATEST(g_test6), ATEST(g_test7),
  ATEST(g_test8), ATEST(g_test9), ATEST(g_test10), ATEST(g_test11),
  ATEST(g_test12), ATEST(g_test13), ATEST(g_test14), ATEST(g_test15),
  ATEST(g_test16), ATEST(g_test17), ATEST(g_test18), ATEST(g_test19),
  ATEST(g_test20), ATEST(g_testQM0), ATEST(g_testQM1)
};
const int testCount = ALEN(g_tests);

void printArray(const int* const a, const int n) {
  printf("{ ");
  for (int i = 0; i < n; i++) {
    printf("%d ", a[i]);
  }
  printf("}");
}

void cmpArrays(void** state, const int32_t* const a, const int32_t* const exp,
               const int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] != exp[i]) {
      printf("\n   expected: ");
      printArray(exp, n);
      printf("\n        got: ");
      printArray(a, n);
      printf("\n");
      fail();
    }
  }
}

void runRoughTest(void** state, int32_t* const a, int32_t* const lr,
                  int32_t* const rl, int32_t* dm, const int rough,
                  const int n) {
  int32_t* const b = new int32_t[n];
  hostLR(a, b, n);
  cmpArrays(state, b, lr, n);

  int32_t* const c = new int32_t[n];
  hostRL(a, c, n);
  cmpArrays(state, c, rl, n);

  int32_t* const d = new int32_t[n];
  hostDM(b, c, d, n);
  cmpArrays(state, d, dm, n);

  assert_int_equal(hostRough(d, n), rough);

  delete[] b;
  delete[] c;
  delete[] d;
}

void runSortTest(void** state, void (*sort)(int32_t* const, const int)) {
  static int32_t a[MAX_TEST_LEN];
  static int32_t expected[MAX_TEST_LEN];

  for (int i = 0; i < testCount; i++) {
    const auto n = g_tests[i].n;
    memcpy(a, g_tests[i].a, sizeof(*a) * n);

    // perform reference sorting for comparison
    memcpy(expected, a, sizeof(*a) * n);
    referenceSort(expected, n);

    sort(a, n);

    cmpArrays(state, a, expected, n);
  }
}

void testDevMemory(void** state) {
  const int n = randLen(1, MAX_TEST_LEN);
  const size_t size = sizeof(int32_t) * n;
  int32_t* const devA = (int32_t* const)cuMalloc(size);
  cuClear(devA, size);

  auto a = new int32_t[n];
  cuDownload(a, devA, size);
  for (int i = 0; i < n; i++) {
    assert_false(a[i]);
  }

  auto b = new int32_t[n];
  randArray(a, n);
  cuUpload(devA, a, size);
  cuDownload(b, devA, size);
  cmpArrays(state, b, a, n);

  cuFree(devA);
  delete[] a;
  delete[] b;
}

void testDevKernel(void** state) {
  int32_t a[] = {1, 2, 3}, expected[] = {1, 4, 9};
  const int n = ALEN(a);
  int32_t* const devA = (int32_t*)cuMalloc(sizeof(a));
  cuUpload(devA, a, sizeof(a));
  devSquare(devA, n);
  cuDownload(a, devA, sizeof(a));

  cmpArrays(state, a, expected, n);

  cuFree(devA);
}

void testReferenceSort(void** state) {
  const int expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int n = ALEN(expected);
  int32_t a[n];
  memcpy(a, g_test19, sizeof(expected));
  referenceSort(a, n);
  cmpArrays(state, a, expected, n);
}

void testHostRough(void** state) {
  runRoughTest(state, g_testQM0, g_testLR0, g_testRL0, g_testDM0, g_testRough,
               ALEN(g_testQM0));
  runRoughTest(state, g_testQM1, g_testLR1, g_testRL1, g_testDM1, g_testRough,
               ALEN(g_testQM1));
}

int32_t g_mergesortBuffer[MAX_TEST_LEN];
void hostMergesortWrap(int32_t* const a, const int n) {
  hostMergesort(a, g_mergesortBuffer, n);
}
void hostMergesortUpWrap(int32_t* const a, const int n) {
  hostMergesortUp(a, g_mergesortBuffer, n);
}

void testHostMergesort(void** state) {
  runSortTest(state, hostMergesortWrap);
}
void testHostMergesortUp(void** state) {
  runSortTest(state, hostMergesortUpWrap);
}
void testHostQuicksort(void** state) {
  runSortTest(state, hostQuicksort);
}
void testHostRoughsort(void** state) {
  runSortTest(state, hostRoughsort);
}
/*
void testDevMergesort(void** state) {
  runSortTest(state, devMergesort);
}
void testDevRoughsort(void** state) {
  runSortTest(state, devRoughsort);
}
*/

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
    cmocka_unit_test(testDevMemory),
    cmocka_unit_test(testDevKernel),
    cmocka_unit_test(testReferenceSort),
    cmocka_unit_test(testHostRough),
    cmocka_unit_test(testHostMergesort),
    cmocka_unit_test(testHostMergesortUp),
    cmocka_unit_test(testHostQuicksort),
    cmocka_unit_test(testHostRoughsort),
  /*cmocka_unit_test(testDevMergesort),
    cmocka_unit_test(testDevRoughsort),*/
    cmocka_unit_test(testXorshift),
  };

  return cmocka_run_group_tests(tests, 0, 0);
}
