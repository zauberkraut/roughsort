/* sort.cxx: Sequential pure-CPU sorting routines. */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "roughsort.h"

void hostMergesort(int32_t* const a, const int n) {
  std::stable_sort(a, a + n);
}

void hostQuicksort(int32_t* const a, const int n) {
  std::sort(a, a + n);
}

/* Generates the LR characteristic sequence of the sequence a. */
void hostLR(const int32_t* const a, int32_t* const b, const int n) {
  b[0] = a[0];
  for (int i = 1; i < n; i++) {
    b[i] = b[i-1] < a[i] ? a[i] : b[i-1];
  }
}

/* Generates the RL characteristic sequence of the sequence a. */
void hostRL(const int32_t* const a, int32_t* const c, const int n) {
  c[n-1] = a[n-1];
  for (int i = n-2; i >= 0; i--) {
    c[i] = c[i+1] > a[i] ? a[i] : c[i+1];
  }
}

/* Generates the disorder measure sequence of the sequence a from above. */
void hostDM(const int32_t* const b, const int32_t* const c, int32_t* const d,
             const int n) {
  int i = n - 1;
  for (int j = n-1; j >= 0; j--) {
    while (j <= i && i >= 0 && c[i] <= b[j] &&
         (j == 0 || c[i] >= b[j-1])) {
      d[i] = i - j;
      i--;
    }
  }
}

/* Computes the smallest k for which a is k-sorted (aka the "radius" of a). */
int hostRough(const int32_t* const d, const int n) {
  int k = INT_MIN;
  for (int i = 0; i < n; i++) {
    if (d[i] > k) {
      k = d[i];
    }
  }
  return k;
}

/* Produces a k-sorted sequence from a (2k + 1)-sorted one. TODO */
void hostHalve(const int32_t* const gamma, int32_t* const delta, const int k,
               const int n) {
  const int r = n / (2*k);

  for (int i = 0; i < r; i++) {

  }
}

/* Sequential roughsort implementation. */
void hostRoughsort(int32_t* const a, const int n) {
}
