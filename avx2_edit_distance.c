#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>
#include <string.h>
#include "src/avx2_edit_distance.h"
#include "src/threadarguments.h"

/*
* n size for each individual block while tiling. 
*/
#define BLOCK_SIZE 512

/*
* Helper macro for diagonal-major indexing on a linearized array.
* Accounts for a width+1 buffer as used in only the avx2 process_tile() 
* function.
*/
#define idx(k, i) ((k) * (height + 1) + (i))

/*
* Helper minimum function for type __m256i (returns whichever of a, b, and c is smaller).
*/
static inline __m256i min_vector(__m256i a, __m256i b, __m256i c){
	__m256i temp = _mm256_min_epi32(a, b);
    return _mm256_min_epi32(temp, c);
}

/*
* Helper minimum function for type int (returns whichever of a, b, and c is smaller).
*/
static inline int min_scalar(int a, int b, int c){
	int temp = (a < b) ? a : b;
	return (temp < c) ? temp : c;
}

/*
* Helper function to process a single tile AND update the global rows 
* and cols.
*/
static int process_tile(int row_offset, int col_offset, 
	int *global_rows_read, int *global_rows_write, int *global_cols, 
	int height, int width, const char *str1, const char *str2, 
	int *buffer, size_t total_len){

	/*
    * Prefill with global rows and cols to allow simpler math
    */
    for(int i = 1; i <= height; i++){
		if(row_offset + i <= total_len){
			buffer[idx(i, i)] = global_cols[row_offset + i];
		}
		else{
			// padding value: not meant to be used
			buffer[idx(i, i)] = total_len;
		}
        
    }
    for(int j = 1; j <= width; j++){
		if(col_offset + j <= total_len){
			buffer[idx(j, 0)] = global_rows_read[col_offset + j];
		}
		else{
			// padding value: not meant to be used
			buffer[idx(j, 0)] = total_len;
		}
    }

	// Initialize the corner separately
	if(col_offset == 0){
		// Grab the value from the left
		buffer[idx(0,0)] = row_offset;
	}
	else if(row_offset == 0){
        // Grab the value from the top
		buffer[idx(0,0)] = col_offset;
	}
	else{
		// Internal tile: grab the value from the top, as normal.
	    buffer[idx(0,0)] = global_rows_read[col_offset];
    }

	const __m256i v_one = _mm256_set1_epi32(1);
	const __m256i v_reverse_idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    /*
    * Calculate all the values inside the tile
    * k is the wavefront number (1-based)
    * Can skip k=1 because it is already initialized
    * i is the row (1-based)
    * j is the column (1-based)
    * We have (j = k - i) <=> (k = j + k) <=> (i = k - j) 
    */ 
    int max_wave = height + width;
	for(int k = 2; k <= max_wave; k++){
		int min_i = (k - width > 1) ? (k - width) : 1;
        int max_i = (k - 1 < height) ? (k - 1) : height;

		// Unrolled (2x) vector loop for performance
		int i;
		for(i = min_i; i <= max_i - 15; i += 16){
			int j = k - i;

			/*
			* Safety check for both vectors--ensure we are not reading 
			* before the start of str2
			*/
			if(col_offset + j - 16 < 0){
				break;
			}

			int *prev_wave = &buffer[idx(k-1, 0)];
			int *prev_wave2 = &buffer[idx(k-2, 0)];

			// Load neighbors for vector 1
			__m256i v_left1 = _mm256_loadu_si256((__m256i*)&prev_wave[i]);
			__m256i v_top1 = _mm256_loadu_si256((__m256i*)&prev_wave[i-1]);
			__m256i v_diag1 = _mm256_loadu_si256((__m256i*)&prev_wave2[i-1]);

			// Load neighbors for vector 2
			__m256i v_left2 = _mm256_loadu_si256((__m256i*)&prev_wave[i+8]);
			__m256i v_top2 = _mm256_loadu_si256((__m256i*)&prev_wave[i-1+8]);
			__m256i v_diag2 = _mm256_loadu_si256((__m256i*)&prev_wave2[i-1+8]);

			/*
			* Load strings
			* vector 1 at (i, j)
			* vector 2 at (i+8, j-8);
			*/
			long long s1c1, s2c1, s1c2, s2c2;
			memcpy(&s1c1, &str1[row_offset + i - 1], 8);
			__m128i v_s1_8_1 = _mm_cvtsi64_si128(s1c1);
			__m256i v_s1_32_1 = _mm256_cvtepi8_epi32(v_s1_8_1);
			memcpy(&s2c1, &str2[col_offset + j - 8], 8);
			__m128i v_s2_8_1 = _mm_cvtsi64_si128(s2c1);
			__m256i v_s2_32_1_fwd = _mm256_cvtepi8_epi32(v_s2_8_1);
			__m256i v_s2_32_1 = _mm256_permutevar8x32_epi32(v_s2_32_1_fwd, v_reverse_idx);

			memcpy(&s1c2, &str1[row_offset + i - 1 + 8], 8);
			__m128i v_s1_8_2 = _mm_cvtsi64_si128(s1c2);
			__m256i v_s1_32_2 = _mm256_cvtepi8_epi32(v_s1_8_2);
			memcpy(&s2c2, &str2[col_offset + j - 8 - 8], 8);
			__m128i v_s2_8_2 = _mm_cvtsi64_si128(s2c2);
			__m256i v_s2_32_2_fwd = _mm256_cvtepi8_epi32(v_s2_8_2);
			__m256i v_s2_32_2 = _mm256_permutevar8x32_epi32(v_s2_32_2_fwd, v_reverse_idx);
			
			/*
			* Find costs
			*/
			// Vector 1
			__m256i v_mask1 = _mm256_cmpeq_epi32(v_s1_32_1, v_s2_32_1);
			// Mask to cost by adding 1 to every component
			__m256i v_cost1 = _mm256_add_epi32(v_mask1, v_one);

			__m256i v_ins1 = _mm256_add_epi32(v_left1, v_one);
			__m256i v_del1 = _mm256_add_epi32(v_top1, v_one);
			__m256i v_sub1 = _mm256_add_epi32(v_diag1, v_cost1);
			__m256i v_res1 = min_vector(v_ins1, v_del1, v_sub1);

			// Vector 2
			__m256i v_mask2 = _mm256_cmpeq_epi32(v_s1_32_2, v_s2_32_2);
			// Mask to cost by adding 1 to every component
			__m256i v_cost2 = _mm256_add_epi32(v_mask2, v_one);

			__m256i v_ins2 = _mm256_add_epi32(v_left2, v_one);
			__m256i v_del2 = _mm256_add_epi32(v_top2, v_one);
			__m256i v_sub2 = _mm256_add_epi32(v_diag2, v_cost2);
			__m256i v_res2 = min_vector(v_ins2, v_del2, v_sub2);

			// Store
			_mm256_storeu_si256((__m256i*)&buffer[idx(k, i)], v_res1);
			_mm256_storeu_si256((__m256i*)&buffer[idx(k, i+8)], v_res2);
		}
		/*
		* Scalar cleanup loop (catches case where we have 1-15 pixels
		* remaining OR edge case in first tile)
		*/
		for(; i <= max_i; i++){
			int j = k - i;
			size_t global_row_idx = row_offset + i;
			size_t global_col_idx = col_offset + j;

			int ins = buffer[idx(k-1, i)] + 1;
			int del = buffer[idx(k-1, i-1)] + 1;
			int cost = (str1[global_row_idx - 1]) == str2[global_col_idx - 1] ? 0 : 1;
			int sub = buffer[idx(k-1-1, i-1)] + cost;
			buffer[idx(k, i)] = min_scalar(ins, del, sub);
		}
	}

	/*
	* Update the global row (write version) so the tile below this 
	* one can read its correct values.
	*/
	for(int j = 1; j <= width; j++){
		global_rows_write[col_offset + j] = buffer[idx(height+j, height)];
	}

    /*
    * Update the global column (write version) so the tile to the 
    * right of this one can read its correct values
    */
    for(int i = 1; i <= height; i++){
        global_cols[row_offset + i] = buffer[idx(i+width, i)];
    }

	return buffer[idx(height+width, height)];
}

/*
* The start_routine function for pthread_create().
*/
static void* thread_worker(void* args){
	/*
	* In this program, args is guaranteed to originally be of 
	* type ThreadArguments. So, casting is safe.
	*/
	ThreadArguments* data = (ThreadArguments*)args;

	/*
	* Create a diagonal-major storage for this thread.
	* Max diagonals = approx 2 * BLOCK_SIZE. Each diagonal needs
	* BLOCK_SIZE elements.
	
	* For BLOCK_SIZE = 512, this is 2 * 513 * 513 * sizeof(int) 
	* = approx 0.5 MB on the heap. For 16 threads, this is ~8 MB.
	* For a system with 16 threads, this should not exhaust the heap.
	* Systems like these tend to have at minimum 16 GB.
	*/
	int *thread_buffer = malloc(2 * (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1) * sizeof(int));
    if(thread_buffer == NULL){
        printf("Unable to allocate memory for a thread's buffer in avx2_edit_distance(). Exiting thread...");
        return NULL;
    }

	// Calculate diagonal bounds
	int n = data->num_blocks;
	int wave = data->current_wave;
	int row_min = (wave < n) ? 0 : (wave - n + 1);
	int row_max = (wave < n) ? wave : (n - 1);

	/*
	* Iterate through every tile in this wave
	*/
	int tile_number = 0;
	for(int r = row_min; r <= row_max; r++){
		// Find the column for this row in the diagonal
		int c = wave - r;

		/*
		* Only process this tile at (r, c) if it is "assigned" to 
		* this thread.
		* Ensures nearly equal spreading of work (eliminating
		* performance bottlenecks in which one thread has more 
		* work than the others).
		*/
		if(tile_number % data->num_threads == data->thread_id){
			size_t row_offset = r * BLOCK_SIZE;
			size_t col_offset = c * BLOCK_SIZE;

			/*
			* Read from (r-1) % 3, write to r % 3 (add 3 to deal with 
			* the negative)
			*/ 
			int read_idx = (r-1+3) % 3;
			int write_idx = r % 3;

			/*
			* Handle edge case where there are not enough elements 
			* left to create a full size tile.
			*/
			int height = (row_offset + BLOCK_SIZE > data->len) 
			? (data->len - row_offset) : BLOCK_SIZE;
			int width = (col_offset + BLOCK_SIZE > data->len)
			? (data->len - col_offset) : BLOCK_SIZE;

			process_tile(row_offset, col_offset,
				data->global_rows[read_idx], data->global_rows[write_idx],
				data->global_cols, height, width, data->str1, 
				data->str2, thread_buffer, data->len);
		}
		tile_number++;
	}

	// Cleanup
	free(thread_buffer);
	return NULL;
}

int avx2_edit_distance(const char *str1, const char *str2, size_t len, long num_threads){
    if(len < 1){
        return 0;
    }
	// Threads limited to 16, should be safe to allocate on stack
	pthread_t threads[num_threads];

    /*
    * sizeof(ThreadArguments) = 80. 80 bytes * 16 max threads should 
    * be safe to allocate on stack
    */ 
	ThreadArguments args[num_threads];

	/*
	* Allocate the global boundaries arrays (holds the numbers future 
	* process_tile calls will need).
	* Triple buffering to guarantee disjoint memory access
	*/
	int *row_bounds[3];
	for(int i = 0; i < 3; i++){
		row_bounds[i] = malloc((len + 1) * sizeof(int));
	}
	int *col_bounds = malloc((len + 1) * sizeof(int));
	if(col_bounds == NULL || row_bounds[0] == NULL ||
		row_bounds[1] == NULL || row_bounds[2] == NULL){
		free(row_bounds[0]);
		free(row_bounds[1]);
		free(row_bounds[2]);
		free(col_bounds);
		return -1;
	}

	/*
	* Initialize the boundaries (row 0 and col 0)
	*/
	for(size_t i = 0; i <= len; i++){
		row_bounds[0][i] = i;
		row_bounds[1][i] = i;
		row_bounds[2][i] = i;
		col_bounds[i] = i;
	}

	/*
	* Loop through the diagonal waves
	*/
	int num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int num_waves = 2 * num_blocks - 1;
	for(size_t wave = 0; wave < num_waves; wave++){
		/*
		* Creating threads to work on the individual tiles 
		* within a wave
		*/
		for(int t = 0; t < num_threads; t++){
			args[t].thread_id = t;
			args[t].num_threads = num_threads;
			args[t].current_wave = wave;
			args[t].num_blocks = num_blocks;

			for(int k = 0; k < 3; k++){
				args[t].global_rows[k] = row_bounds[k];
			}
			args[t].global_cols = col_bounds;
			args[t].len = len;
			args[t].str1 = str1;
			args[t].str2 = str2;

			// Launch the thread with its appropriate arguments
			pthread_create(&threads[t], NULL, thread_worker, &args[t]);
		}

		/*
		* Ensure synchronization. This wave should be finished before
		* moving to the next.
		*/
		for(int t = 0; t < num_threads; t++){
			pthread_join(threads[t], NULL);
		}
	}

	// Cleanup
	int result = row_bounds[(num_blocks - 1) % 3][len];
	
	for(int i = 0; i < 3; i++){
		free(row_bounds[i]);
	}
	free(col_bounds);
	return result;
}