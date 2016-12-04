/* sort.cxx: Sequential pure-CPU sorting routines. */

#include <algorithm>
#include "roughsort.h"

namespace {

/* Generates the LR characteristic sequence of the array a. */
void hostLR(const int32_t* const a, int32_t* const b, const int n) {
  b[0] = a[0];
  for (int i = 1; i < n; i++) {
    b[i] = b[i-1] < a[i] ? a[i] : b[i-1];
  }
}

/* Generates the RL characteristic sequence of the array a. */
void hostRL(const int32_t* const a, int32_t* const c, const int n) {
  c[n-1] = a[n-1];
  for (int i = n-2; i >= 0; i--) {
    c[i] = c[i+1] > a[i] ? a[i] : c[i+1];
  }
}

/* Generates the disorder measure sequence of the array a and returns its
   radius. */
int hostDM(const int32_t* const b, const int32_t* const c, const int n) {
  int radius = 0;

  int i = n - 1;
  for (int j = n-1; j >= 0; j--) {
    while (j <= i && i >= 0 && c[i] <= b[j] &&
           (j == 0 || c[i] >= b[j-1])) {
      radius = std::max(radius, i - j);
      i--;
    }
  }

  return radius;
}

void oneSort(int32_t* const a, const int n) {
  int32_t prev = a[0];

  for (int i = 1; i < n; i++) {
    int32_t x = a[i];
    if (x < prev) {
      a[i] = prev;
      a[i-1] = x;
    } else {
      prev = x;
    }
  }
}

/* Produces a (k - 1)/2-sorted sequence from a k-sorted one. */
void hostHalveEasy(int32_t* const a, const int radius, const int n) {
  const int mid = radius >> 1;
  int32_t* const endSeg = a + n - radius;

  for (int32_t* offset = a; offset <= endSeg; offset += radius) {
    std::nth_element(offset, offset + mid, offset + radius);
  }

  for (int32_t* offset = a + mid; offset <= endSeg; offset += radius) {
    std::nth_element(offset, offset + mid, offset + radius);
  }

  for (int32_t* offset = a; offset <= endSeg; offset += radius) {
    std::nth_element(offset, offset + mid, offset + radius);
  }
}

/* Produces a (k - 1)/2-sorted sequence from a k-sorted one. */
void hostHalve(int32_t* const a, const int radius, const int n) {
  const int mid = radius >> 1;
  int32_t* const end = a + n;
  const int tailLen = n%radius;
  int32_t* const endSeg = end - tailLen - radius;
  const int tailMid = tailLen/2;

  int32_t* offset;
  for (offset = a; offset <= endSeg; offset += radius) {
    std::nth_element(offset, offset + mid, offset + radius);
  }
  std::nth_element(offset, offset + tailMid, end);

  for (offset = a + mid; offset <= endSeg; offset += radius) {
    std::nth_element(offset, offset + mid, offset + radius);
  }
  if (tailLen) {
    std::nth_element(offset, end - tailLen, end);
  }

  for (offset = a; offset <= endSeg; offset += radius) {
    std::nth_element(offset, offset + mid, offset + radius);
  }
  std::nth_element(offset, offset + tailMid, end);
}

} // end anonymous namespace

void hostQuicksort(int32_t* const a, const int n) {
  std::sort(a, a + n);
}

void hostBubblesort(int32_t* const a, const int n) {
  bool outOfOrder;

  do {
    outOfOrder = false;

    for (int i = 0; i < n-1; i++) {
      int l = a[i], r = a[i+1];
      if (l > r) {
        a[i] = r;
        a[i+1] = l;
        outOfOrder = true;
      }
    }
  } while (outOfOrder);
}

/* Computes the smallest k for which a is k-sorted (aka the "radius" of a). */
int hostRadius(const int32_t* const a, const int n) {
  int32_t* b = new int[n];
  int32_t* c = new int[n];
  hostLR(a, b, n);
  hostRL(a, c, n);
  hostDM(b, c, n);

  const int k = hostDM(b, c, n);

  delete[] b;
  delete[] c;
  return k;
}

/* Sequential roughsort implementation. */
void hostRoughsort(int32_t* const a, const int n) {
  const int radius = hostRadius(a, n);

  if (!radius || n < 2) {
    return;
  }
  if (radius == 1) {
    oneSort(a, n);
    return;
  }

  typedef void (*halveFunc)(int32_t* const, const int, const int);
  halveFunc halve = n%radius ? (halveFunc)hostHalve : (halveFunc)hostHalveEasy;

  int k = radius, p = 0;
  do {
    halve(a, k, n);
    k = radius / (2 << p++);
  } while (k > 1);
}
