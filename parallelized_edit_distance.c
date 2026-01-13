#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include "src/parallelized_edit_distance.h"
#include "src/threadarguments.h"

/*
* n size for each individual block while tiling. 
* See derivation in "lab7 block size calculation.pdf".
*/
#define BLOCK_SIZE 512

/*
* Helper minimum function (returns whichever of a, b, and c is smaller).
*/
static inline int min(int a, int b, int c){
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
	int *prev_row, int* cur_row){

	/*
	* Set up prev_row
	*/
	for(int j = 1; j <= width; j++){
		prev_row[j] = global_rows_read[col_offset + j];
	}

	// Initialize the corner separately
	if(col_offset == 0){
		// Grab the value from the left
		prev_row[0] = row_offset;
	}
	else if(row_offset == 0){
		prev_row[0] = col_offset;
	}
	else{
		// Internal tile: grab the value from the top, as normal.
		prev_row[0] = global_rows_read[col_offset];
	}

	for(int i = 1; i <= height; i++){
		size_t global_row_index = row_offset + i;
		cur_row[0] = global_cols[global_row_index];

		for(int j = 1; j <= width; j++){
			size_t global_col_index = col_offset + j;

			int cost;
			if(str1[global_row_index - 1] == str2[global_col_index - 1]){
				cost = prev_row[j - 1];
			}
			else{
				int deletion = prev_row[j] + 1;
				int insertion = cur_row[j-1] + 1;
				int substitution = prev_row[j-1] + 1;
				cost = min(deletion, insertion, substitution);
			}
			cur_row[j] = cost;
		}

		/*
		* Update the global column so the tile to the right of this one can read its
		* correct initial cur_row[0]
		*/
		global_cols[global_row_index] = cur_row[width];

		/*
		* Swap pointers; next iteration's prev_row is this 
		* iteration's cur_row.
		*/
		int *temp = prev_row;
		prev_row = cur_row;
		// Don't really need to keep track of this
		cur_row = temp;
	}

	/*
	* Update the global row (write version) so the tile below this 
	* one can read its correct initial prev_row values.
	*/
	for(int j = 1; j <= width; j++){
		global_rows_write[col_offset + j] = prev_row[j];
	}

	return prev_row[width];
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
	* Create a prev_row and cur_row for this thread to use for 
	* each tile
	*/
	int *thread_prev_row = malloc((BLOCK_SIZE + 1) * sizeof(int));
	int *thread_cur_row = malloc((BLOCK_SIZE + 1) * sizeof(int));

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
				data->str2, thread_prev_row, thread_cur_row);
		}
		tile_number++;
	}

	// Cleanup
	free(thread_prev_row);
	free(thread_cur_row);
	return NULL;
}

int parallelized_edit_distance(const char *str1, const char *str2, size_t len, long num_threads){
	if(len < 1){
        return 0;
    }
    pthread_t threads[num_threads];
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