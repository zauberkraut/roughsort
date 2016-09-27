# roughsort
Parallel CUDA sorter of k-sorted sequences.

---

Dependencies:
  * CUDA SDK 8.0+
  * GCC 4.6+
  * GNU Make
  * CMocka for unit tests

---

To build:  
`make`

To build with debugging symbols:  
`make debug`

To run unit tests:  
`make test`

For usage instructions, run (after building):  
`roughsort -h`

To sort a randomly-generated array of random length, run:  
`roughsort`

To sort a randomly-generated array of 20000 integers, run:  
`roughsort -n 20000`

To compare the sorting results to a reference implementation, add the -t flag:  
`roughsort -tn 20000`

No CUDA kernels are implemented yet.
