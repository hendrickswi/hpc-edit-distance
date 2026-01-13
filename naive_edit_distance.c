#include <stdlib.h>
#include <stdio.h>
#include "src/naive_edit_distance.h"

/*
* Helper minimum function (returns whichever of a, b, and c is smaller).
*/
static inline int min(int a, int b, int c){
	int temp = (a < b) ? a : b;
	return (temp < c) ? temp : c;
}

int naive_edit_distance(const char *str1, const char *str2, size_t len){
    if(len < 1){
        return 0;
    }
	int *prev_row = malloc((len + 1) * sizeof(int));
	int *cur_row = malloc((len + 1) * sizeof(int));

	/*
	* Check for allocation errors.
	*/
	if(prev_row == NULL || cur_row == NULL){
		printf("Failed to allocate memory for row arrays in naive_edit_distance(), exiting program.");
		free(prev_row);
		free(cur_row);
		return -1;
	}

	/*
	* Set up prev_row
	*/
	for(int i = 0; i <= len; i++){
		prev_row[i] = i;
	}

	for(int i = 1; i <= len; i++){
		cur_row[0] = i;
		/*
		* Calculate costs for the current row from neighbors.
		*/
		for(int j = 1; j <= len; j++){
			int cost;
			if(str1[i-1] == str2[j-1]){
				cost = prev_row[j-1];
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
		* Swap pointers; next iteration's prev_row is this iteration's cur_row.
		*/
		int *temp = prev_row;
		prev_row = cur_row;
		
		// Don't really need to keep track of this
		cur_row = temp;
	}

	/*
	* The edit distance is always the bottom right number in the table.
	* Note that prev_row contains the last row after the swap.
	*/
	int result = prev_row[len];
	free(prev_row);
	free(cur_row);
	return result;
}