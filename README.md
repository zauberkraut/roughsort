# roughsort
Parallel CUDA sorter of k-sorted sequences.

---

Dependencies:
  * CUDA SDK 8.0
  * GCC
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
`roughsort`

To sort a randomly-generated array of 20000 integers, run:  
`roughsort -c -n 20000`

No CUDA kernels are implemented yet, so the binary must be run with the '-c' flag to disable GPU computation.
