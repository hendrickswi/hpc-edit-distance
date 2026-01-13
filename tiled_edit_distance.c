#include <stdlib.h>
#include <stdio.h>
#include "src/tiled_edit_distance.h"

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
* Helper function to process a single tile AND update the global rows and cols.
* Returns -1 if memory could not be allocated for the two rows.
* Returns 0 otherwise.
*/
static int process_tile(int row_offset, int col_offset, int *global_rows_read, 
	int* global_rows_write, int *global_cols, int height, int width, 
	const char *str1, const char *str2){

	int *prev_row = malloc((width + 1) * sizeof(int));
	int *cur_row = malloc((width + 1) * sizeof(int));

	/*
	* Check for allocation errors.
	*/
	if(prev_row == NULL || cur_row == NULL){
		printf("Failed to allocate memory for row arrays in tiled_edit_distance(), exiting program.");
		free(prev_row);
		free(cur_row);
		return -1;
	}

	/*
	* Set up prev_row using global_rows_read
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
		/*
		* Safe read because neighboring tile to the left writes to 
		* global_rows_write, not global_rows_read.
		*/
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
		* Swap pointers; next iteration's prev_row is this iteration's 
		* cur_row.
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

	// Cleanup
	int result = prev_row[width];
	free(prev_row);
	free(cur_row);
	return result;
}

int tiled_edit_distance(const char *str1, const char *str2, size_t len){
    if(len < 1){
        return 0;
    }
	/*
	* Allocate the global boundaries arrays (holds the numbers future 
	* process_tile calls will need).
	*/
	int *row_bounds1 = malloc((len + 1) * sizeof(int));
	int *row_bounds2 = malloc((len + 1) * sizeof(int));
	int *col_bounds = malloc((len + 1) * sizeof(int));

	/*
	* Check for allocation errors
	*/
	if(row_bounds1 == NULL || row_bounds2 == NULL || 
		col_bounds == NULL){

		free(row_bounds1);	
		free(row_bounds2);
		free(col_bounds);
		return -1;
	}

	/*
	* Initialize the boundaries (row 0 and col 0)
	*/
	for(size_t i = 0; i <= len; i++){
		row_bounds1[i] = i;
		row_bounds2[i] = i;
		col_bounds[i] = i;
	}

	/*
	* Sequentially iterate through tiles, row-by-row.
	*/
	for(size_t i = 0; i < len; i += BLOCK_SIZE){
		/*
		* To avoid off-by-one errors caused by shared boundary arrays,
		* use double buffering.
		*/
		int block_row_index = i / BLOCK_SIZE;
		int *current_read_buffer, *current_write_buffer;

		if(block_row_index % 2 == 0){
			// Even rows
			current_read_buffer = row_bounds1;
			current_write_buffer = row_bounds2;
		}
		else{
			// Odd rows
			current_read_buffer = row_bounds2;
			current_write_buffer = row_bounds1;
		}

		for(size_t j = 0; j < len; j+= BLOCK_SIZE){
			/*
			* Handle edge case where there are not enough elements left to create a full size tile.
			*/
			int height = (i + BLOCK_SIZE > len) ? (len - i) : BLOCK_SIZE;
			int width = (j + BLOCK_SIZE > len) ? (len - j) : BLOCK_SIZE;

			int result = process_tile(i, j, current_read_buffer, current_write_buffer, 
				col_bounds, height, width, str1, str2);
			if(result == -1){
				free(row_bounds1);
				free(row_bounds2);
				free(col_bounds);
				return result;
			}
		}
	}

	// Cleanup
	int num_block_rows = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int result;
	if((num_block_rows - 1) % 2 == 0){
		// Last row was even
		result = row_bounds2[len];
	}
	else{
		// Last row was odd
		result = row_bounds1[len];
	}

	free(row_bounds1);
	free(row_bounds2);
	free(col_bounds);
	return result;
}