/* sort.cxx: Sequential pure-CPU sorting routines. */

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

inline void swap(int32_t* const a, const int i, const int j) {
  const auto l = a[i], r = a[j];
  a[i] = r;
  a[j] = l;
}

void mergesort(int32_t* const a, int32_t* const b, const int n,
               const int depth) {
  const auto llen = n / 2, rlen = n - llen;

  if (depth) {
    if (llen > 1) {
      mergesort(b, a, llen, depth - 1);
    }
    if (rlen > 1) {
      mergesort(b + llen, a + llen, rlen, depth - 1);
    }

    for (int l = 0, r = llen, k = l; k < n; k++) { // merge
      auto x = (r == n || (l != llen && b[l] <= b[r])) ? b[l++] : b[r++];
      a[k] = x;
    }
  } else if (a[0] > a[1]) { // order bottom-level pair in place
    swap(a, 0, 1);
  }
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

/* qsort() from cstdlib, used for reference. */
void referenceSort(int32_t* const a, const int n) {
  qsort(a, n, sizeof(*a),
        [](const void* x, const void* y) {
          // x - y can cause signed int overflow, whose behavior is undefined
          return *(int32_t*)x < *(int32_t*)y ? -1 :
                 (*(int32_t*)x > *(int32_t*)y ? 1 : 0);
        });
}

/* Linear-space top-down mergesort. */
void hostMergesort(int32_t* const a, int32_t* const b, const int n) {
  if (n > 1) {
    const auto depth = (int)floor(log2(n));
    bool sortToB = depth & 1;
    mergesort(sortToB ? b : a, sortToB ? a : b, n, depth);
    if (sortToB) { // sort back to a
      memcpy(a, b, sizeof(*a) * n);
    }
  }
}

/* Linear-space bottom-up mergesort. */
void hostMergesortUp(int32_t* const a, int32_t* const b, const int n) {
  int32_t *src = a, *dst = b;

  for (int runLen = 1; runLen < n; runLen <<= 1) {
    const int mergeCount = (n + runLen - 1) / runLen >> 1;

    for (int merge = 0; merge < mergeCount; merge++) {
      int l = (merge << 1) * runLen, r = l + runLen;
      const int lend = r, rend = r + runLen < n ? r + runLen : n;

      for (int k = l; k < rend; k++) {
        auto x = (r == rend || (l != lend && src[l] <= src[r])) ?
                 src[l++] : src[r++];
        dst[k] = x;
      }
    }

    const int tail = (runLen << 1) * mergeCount;
    if (tail < n) {
      memcpy(dst + tail, src + tail, sizeof(int32_t) * (n - tail));
    }

    int32_t* tmp = src;
    src = dst;
    dst = tmp;
  }

  if (src != a) {
    memcpy(a, b, sizeof(int32_t) * n);
  }
}

/* A somewhat-lousy quicksort implementation. */
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

/* Sequential roughsort implementation. */
void hostRoughsort(int32_t* const a, const int n) {
}
