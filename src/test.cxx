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
        g_test21[] = {2, 3, 5, 1, 4, 2, 6, 8, 7, 9, 8, 11, 6, 13, 12,
                      16, 15, 17, 18, 20, 18, 19, 21, 19, 30, 31, 32, 33, 34},
        g_test21LR[] = {2, 3, 5, 5, 5, 5, 6, 8, 8, 9, 9, 11, 11, 13, 13,
                        16, 16, 17, 18, 20, 20, 20, 21, 21, 30, 31, 32, 33, 34},
        g_test21RL[] = {1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6, 6, 12, 12, 15,
                        15, 17, 18, 18, 18, 19, 19, 19, 30, 31, 32, 33, 34},
        g_test21DM[] = {0, 1, 2, 3, 3, 4, 0, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1,
                        0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0},
        g_test22[] = {2, 3, 5, 1, 4, 2, 6, 8, 7, 9, 8, 11, 6, 13, 12,
                      16, 15, 17, 18, 20, 18, 19, 21, 19},
        g_test22LR[] = {2, 3, 5, 5, 5, 5, 6, 8, 8, 9, 9, 11, 11, 13, 13,
                        16, 16, 17, 18, 20, 20, 20, 21, 21},
        g_test22RL[] = {1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6, 6, 12, 12, 15,
                        15, 17, 18, 18, 18, 19, 19, 19},
        g_test22DM[] = {0, 1, 2, 3, 3, 4, 0, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1,
                        0, 0, 0, 1, 2, 3, 4};
const int g_testRadius = 5;

#define ALEN(a) (sizeof(a) / sizeof(*a))
#define ATEST(a) {a, ALEN(a)}
struct Test { int32_t* const a; const int n; } g_tests[] = {
  ATEST(g_test0), ATEST(g_test1), ATEST(g_test2), ATEST(g_test3),
  ATEST(g_test4), ATEST(g_test5), ATEST(g_test6), ATEST(g_test7),
  ATEST(g_test8), ATEST(g_test9), ATEST(g_test10), ATEST(g_test11),
  ATEST(g_test12), ATEST(g_test13), ATEST(g_test14), ATEST(g_test15),
  ATEST(g_test16), ATEST(g_test17), ATEST(g_test18), ATEST(g_test19),
  ATEST(g_test20), ATEST(g_test21), ATEST(g_test22)
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
      printf("\n  expected: ");
      printArray(exp, n);
      printf("\n       got: ");
      printArray(a, n);
      printf("\n");
      fail();
    }
  }
}

void runRoughTest(void** state, int32_t* const a, int32_t* const lr,
                  int32_t* const rl, int32_t* dm, const int rough,
                  const int n) {
  assert_int_equal(hostRadius(a, n), rough);
}

void runHostSortTest(void** state, void (*sort)(int32_t* const, const int)) {
  static int32_t a[MAX_TEST_LEN];
  static int32_t expected[MAX_TEST_LEN];

  for (int i = 0; i < testCount; i++) {
    const auto n = g_tests[i].n;
    memcpy(a, g_tests[i].a, 4*n);

    // perform reference sorting for comparison
    memcpy(expected, a, 4*n);
    hostQuicksort(expected, n);

    sort(a, n);

    cmpArrays(state, a, expected, n);
  }
}

void runDevSortTest(void** state, void (*sort)(int32_t* const, const int)) {
  static int32_t hostA[MAX_TEST_LEN];
  int32_t* const devA =
    static_cast<int32_t* const>(cuMalloc(4 * MAX_TEST_LEN));
  static int32_t expected[MAX_TEST_LEN];

  for (int i = 0; i < testCount; i++) {
    const auto n = g_tests[i].n;
    cuUpload(devA, g_tests[i].a, 4*n);

    // perform reference sorting for comparison
    memcpy(expected, g_tests[i].a, 4*n);
    hostQuicksort(expected, n);

    sort(devA, n);

    cuDownload(hostA, devA, 4*n);
    cmpArrays(state, hostA, expected, n);
  }

  cuFree(devA);
}

void testDevMemory(void** state) {
  const int n = randLen(1, MAX_TEST_LEN);
  const size_t size = 4 * n;
  int32_t* const devA = (int32_t* const)cuMalloc(size);
  cuClear(devA, size);

  auto a = new int32_t[n];
  cuDownload(a, devA, size);
  for (int i = 0; i < n; i++) {
    assert_false(a[i]);
  }

  auto b = new int32_t[n];
  randArray(a, -1, n, false);
  cuUpload(devA, a, size);
  cuDownload(b, devA, size);
  cmpArrays(state, b, a, n);

  cuFree(devA);
  delete[] a;
  delete[] b;
}

void testHostRough(void** state) {
  runRoughTest(state, g_test21, g_test21LR, g_test21RL, g_test21DM,
               g_testRadius, ALEN(g_test21));
  runRoughTest(state, g_test22, g_test22LR, g_test22RL, g_test22DM,
               g_testRadius, ALEN(g_test22));
}

void testHostQuicksort(void** state) {
  runHostSortTest(state, hostQuicksort);
}
void testHostBubblesort(void** state) {
  runHostSortTest(state, hostBubblesort);
}
void testHostRoughsort(void** state) {
  runHostSortTest(state, hostRoughsort);
}
void testDevMergesort(void** state) {
  runDevSortTest(state, devMergesort);
}
void testDevRadixsort(void** state) {
  runDevSortTest(state, devRadixsort);
}
void testDevRoughsort(void** state) {
  // hack
  auto wrapDevRoughsort = [](int32_t* const a, const int n) {
    devRoughsort(a, MAX_TEST_LEN, n);
  };
  runDevSortTest(state, wrapDevRoughsort);
}

} // end anonymous namespace

int main() {
  randInit(false);

  const struct CMUnitTest tests[] = {
    cmocka_unit_test(testDevMemory),
    cmocka_unit_test(testHostRough),
    cmocka_unit_test(testHostQuicksort),
    cmocka_unit_test(testHostBubblesort),
    cmocka_unit_test(testHostRoughsort),
    cmocka_unit_test(testDevMergesort),
    cmocka_unit_test(testDevRadixsort),
    cmocka_unit_test(testDevRoughsort),
  };

  return cmocka_run_group_tests(tests, 0, 0);
}
