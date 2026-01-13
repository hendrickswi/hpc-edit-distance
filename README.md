The goal of this project was to create an optimized version of the traditional Levenshtein edit distance algorithm. This "naive" version is present in naive_edit_distance.c. 
The first optimization attempted is present in tiled_edit_distance.c, which uses tiling/blocking in an attempt to increase cache hits, thereby leading to more performance.
The next optimization attempted is present in parallelized_edit_distance.c, which uses pthread multithreading to drastically increase performance. Builds on the tiled_edit_distance() algorithm by allowing each thread to work on one independent tile at a time, as opposed to working single-threaded.
The final series of optimizations are present in avx2_edit_distance.c, which utilizies AVX2 SIMD instructions to compute vectors of eight values at once. This implementation also features loop unrolling and a diagonal-major optimization, which lends itself particularly well to vectorization due to placing dependent data contiguously 
in memory, increasing cache locality.

In the end, these optimizations achieved a 38x speedup on large datasets of 500k characters compared to the baseline, which improves further as datasets grow.
main.c is essentially a benchmarking tool used to verify correctness (compared to the known naive implementation) and show performance.
