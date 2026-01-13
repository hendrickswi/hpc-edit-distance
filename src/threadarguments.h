#include <stddef.h>

/*
* Struct to hold arguments threads will pass to process_tile()
*/
typedef struct {
		int thread_id;
		long num_threads;
		int current_wave;
		int num_blocks;
		int *global_rows[3];
		int *global_cols;
		size_t len;
		const char *str1;
		const char *str2;
	} ThreadArguments;