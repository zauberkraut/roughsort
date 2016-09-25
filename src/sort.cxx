/* sort.cxx: Sequential pure-CPU sorting routines. */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "roughsort.h"

namespace {

inline void swap(int* const a, const int i, const int j) {
  const int l = a[i], r = a[j];
  a[i] = r;
  a[j] = l;
}

void mergesort(int* const a, int* const b, const int i, const int n,
               const int depth) {
  const int llen = n / 2,
            rlen = n - llen,
            mid  = i + llen;

  if (depth) {
    if (llen > 1) {
      mergesort(b, a, i, llen, depth - 1);
    }
    if (rlen > 1) {
      mergesort(b, a, mid, rlen, depth - 1);
    }
  }

  if (!depth) {
    if (a[i] > a[mid]) { // swap src pairs in place
      swap(a, i, mid);
    }
  } else { // merge
    const int end = i + n;
    for (int l = i, r = mid, k = l; k < end; k++) {
      int x = (r == end || (l != mid && b[l] <= b[r])) ? b[l++] : b[r++];
      a[k] = x;
    }
  }
}

/* Selects the median of the first, middle and last elements of the array. */
int median3(const int* const a, int* const pivot, const int n) {
  int i = n/2, j = n-1, k;
  int lo = a[0], mid = a[i], hi = a[j];

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

/* Mergesort launcher */
void hostMergesort(int* const a, int* const b, const int n) {
  if (n > 1) {
    auto depth = (int)floor(log2(n));
    bool sortToB = depth & 1;
    mergesort(sortToB ? b : a, sortToB ? a : b, 0, n, depth);
    if (sortToB) { // sort back to a
      memcpy(a, b, sizeof(*a) * n);
    }
  }
}

/* Quicksort from cstdlib. */
void hostQuicksortC(int* const a, const int n) {
  qsort(a, n, sizeof(int),
        [](const void* x, const void* y) {
          // x - y can cause signed int overflow, whose behavior is undefined
          return *(int*)x < *(int*)y ? -1 : (*(int*)x > *(int*)y ? 1 : 0);
        });
}

/* For comparison with the indirection of cstdlib:qsort().
   Currently performs terribly over already-sorted arrays. */
void hostQuicksort(int* const a, const int n) {
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
