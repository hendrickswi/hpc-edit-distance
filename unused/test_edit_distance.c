#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "naive_edit_distance.h"
#include "../tiled_edit_distance.h"
#include "parallelized_edit_distance.h"
#include "../avx2_edit_distance.h"

#define NUM_THREADS 4
#define BLOCK_SIZE 512

static void run_test_expected(const char* name, const char* s1, const char* s2, int expected) {
    printf("Running Test: %s ... ", name);
    fflush(stdout);
    
    size_t len = strlen(s1);

    // Run implementations
    int r_naive = naive_edit_distance(s1, s2, len);
    int r_tiled = tiled_edit_distance(s1, s2, len);
    int r_para  = parallelized_edit_distance(s1, s2, len, NUM_THREADS);
    int r_avx   = avx2_edit_distance(s1, s2, len, NUM_THREADS);

    // Check against expected
    int fail = 0;
    if (r_naive != expected) { fail = 1; }
    if (r_tiled != expected) { fail = 1; }
    if (r_para != expected)  { fail = 1; }
    if (r_avx != expected)   { fail = 1; }

    // Print results
    if (fail) {
        printf("FAILED!\n");
        printf("Naive:    %d\n", r_naive);
        printf("Tiled:    %d\n", r_tiled);
        printf("Parallel: %d\n", r_para);
        printf("AVX2:     %d\n", r_avx);
    } else {
        printf("PASSED (Result: %d)\n", r_naive);
    }
    fflush(stdout);
}

static void run_test_naive(const char* name, const char* s1, const char* s2) {
    printf("Running Test: %s ... ", name);
    fflush(stdout);
    
    size_t len = strlen(s1);

    // Run implementations
    int r_naive = naive_edit_distance(s1, s2, len);
    int r_tiled = tiled_edit_distance(s1, s2, len);
    int r_para  = parallelized_edit_distance(s1, s2, len, NUM_THREADS);
    int r_avx   = avx2_edit_distance(s1, s2, len, NUM_THREADS);

    // Check against naive implementation
    int fail = 0;
    if (r_tiled != r_naive) { fail = 1; }
    if (r_para != r_naive)  { fail = 1; }
    if (r_avx != r_naive)   { fail = 1; }

    // Print results
    if (fail) {
        printf("FAILED!\n");
        printf("Naive:    %d\n", r_naive);
        printf("Tiled:    %d\n", r_tiled);
        printf("Parallel: %d\n", r_para);
        printf("AVX2:     %d\n", r_avx);
    } else {
        printf("PASSED (Result: %d)\n", r_naive);
    }
    fflush(stdout);
}

// Helper to generate long strings
static char* make_string(size_t len, char fill) {
    char* str = malloc(len + 1);
    for(size_t i=0; i<len; i++) str[i] = fill;
    str[len] = '\0';
    return str;
}

int main(){
    /*
    * Simple tests
    */
    run_test_expected("Empty Strings", "", "", 0);
    run_test_expected("Simple Match", "HELLO", "HELLO", 0);
    run_test_expected("Simple Sub", "HELLO", "HELLA", 1);
    run_test_expected("Simple Del", "HELLO", "HELL", 1);
    run_test_expected("Completely Different", "ABC", "XYZ", 3);

    /*
    * Block boundary tests
    */
    char* s63_a = make_string(BLOCK_SIZE-1, 'A');
    char* s63_b = make_string(BLOCK_SIZE-1, 'B');
    run_test_expected("1 Under Block Size", s63_a, s63_b, BLOCK_SIZE-1);

    char* s64_a = make_string(BLOCK_SIZE, 'A');
    char* s64_b = make_string(BLOCK_SIZE, 'B');
    run_test_expected("Exact Block Size", s64_a, s64_b, BLOCK_SIZE);

    char* s65_a = make_string(BLOCK_SIZE+1, 'A');
    char* s65_b = make_string(BLOCK_SIZE+1, 'B');
    run_test_expected("1 Over Block Size", s65_a, s65_b, BLOCK_SIZE+1);

    /*
    * Race condition tests
    */
    // If Tile(1,1) reads a result from Tile(0,0), cost will be > 0.
    char* s_long = make_string(2048, 'X');
    run_test_expected("Large Identical (Race Condition Check)", s_long, s_long, 0);

    // Cleanup
    free(s63_a); free(s63_b);
    free(s64_a); free(s64_b);
    free(s65_a); free(s65_b);
    free(s_long);
    return 0;
}