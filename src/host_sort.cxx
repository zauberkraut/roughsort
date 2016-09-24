#include <cstdlib>
#include <cstring>
#include "roughsort.h"

namespace {

void mergesort(int* const a, int* const b, const int i, const int n) {
  const int llen = n/2, rlen = n - llen, mid = i + llen;

  if (llen >= 2) {
    mergesort(a, b, i, llen);
  }
  if (rlen >= 2) {
    mergesort(a, b, mid, rlen);
  }

  // merge
  memcpy(b + i, a + i, sizeof(*a) * n);
  const int end = i + n;
  for (int l = i, r = mid, k = l; k < end; k++) {
    int x = (r == end || (l != mid && b[l] <= b[r])) ? b[l++] : b[r++];
    a[k] = x;
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

void hostMergesort(int* const a, const int n) {
  // TODO: replace this function with a closure over b to speed it up
  auto b = new int[n]; // don't allocate GiB on the stack
  mergesort(a, b, 0, n);
  delete[] b;
}

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

  auto tmp = a[k];
  a[k] = a[r];
  a[r] = tmp;

  auto i = 0;
  for (auto j = 0; j < r; j++) {
    auto x = a[j];
    if (x < pivot) {
      tmp = a[i];
      a[i] = x;
      a[j] = tmp;
      i++;
    }
  }

  tmp = a[i];
  a[i] = a[r];
  a[r] = tmp;

  hostQuicksort(a, i);
  hostQuicksort(a + i + 1, n - i - 1);
}
