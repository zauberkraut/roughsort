/* sort.cxx: Sequential pure-CPU sorting routines. */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "roughsort.h"

namespace {

inline void swap(int32_t* const a, const int i, const int j) {
  const auto l = a[i], r = a[j];
  a[i] = r;
  a[j] = l;
}

/* Selects the median of the first, middle and last elements of the array. */
int median3(const int32_t* const a, int32_t* const pivot, const int n) {
  int i = n/2, j = n-1, k;
  const auto lo = a[0], mid = a[i], hi = a[j];

  if (mid <= lo && lo <= hi) {
    k = 0; *pivot = lo;
  } else if (lo <= mid && mid <= hi) {
    k = i; *pivot = mid;
  } else {
    k = j; *pivot = hi;
  }

  return k;
}

} // end anonymous namespace

/* Linear-space, partially-optimized bottom-up mergesort. */
void hostMergesort(int32_t* const a, int32_t* const b, const int n) {
  auto *src = a, *dst = b;

  for (int runLen = 1; runLen < n; runLen <<= 1) {
    int l = 0;
    for (int r = runLen; r < n; l = r, r += runLen) {
      const auto lend = r, rend = r + runLen < n ? r + runLen : n;

      for (int k = l; k < rend; k++) {
        auto x = (r == rend || (l != lend && src[l] <= src[r])) ?
                 src[l++] : src[r++];
        dst[k] = x;
      }
    }

    if (l < n) {
      memcpy(dst + l, src + l, sizeof(int32_t) * (n - l));
    }

    auto* tmp = src;
    src = dst;
    dst = tmp;
  }

  if (src != a) {
    memcpy(a, b, sizeof(int32_t) * n);
  }
}

/* A simple quicksort implementation. */
void hostQuicksort(int32_t* const a, const int n) {
  if (n < 2) {
    return;
  }

  int pivot;
  const auto k = median3(a, &pivot, n); // k = pivot index in a
  const auto r = n - 1;

  swap(a, k, r);

  auto i = 0;
  for (auto j = 0; j < r; j++) {
    if (a[j] < pivot) {
      swap(a, i++, j);
    }
  }

  swap(a, i, r);

  hostQuicksort(a, i);
  hostQuicksort(a + i + 1, n - i - 1);
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
