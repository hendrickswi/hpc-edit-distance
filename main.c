#define _POSIX_C_SOURCE_199309L
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "naive_edit_distance.h"
#include "src/tiled_edit_distance.h"
#include "src/parallelized_edit_distance.h"
#include "src/avx2_edit_distance.h"

static const size_t n = 100000;
static const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";


static double returnCurrentTime() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1000 + (double)ts.tv_nsec / 1e6;
}

char *generate_random_str(size_t len){
	char *str = malloc(len + 1);
	
	/*
	* Check for allocation error
	*/
	if(str == NULL){
		printf("Unable to initialize a string in generate_random_str(size_t len).");
		return NULL;
	}

	/*
	* Construct the string
	*/
	for(size_t i = 0; i < len; i++){
		// Get an integer on the interval [0, sizeof(charset))
		int select = rand() % (int)(sizeof(charset) - 1);
		str[i] = charset[select];
	}
	str[len] = '\0';
	return str;
}

int main(){
	const long num_processors = sysconf(_SC_NPROCESSORS_ONLN);
	/*
	* Fallback: 2 threads is a safe estimate for modern computers 
	* (last 15-20 years) to avoid constant context switching.
	* This is in case the sysconf() call fails for some reason.
	
    * Fallback: 16 threads is the max amount this program will use
    * due to needing to balance parallelism and overhead from creating
	* threads.
	*/
    int num_threads;
	if(num_processors < 1) { num_threads = 2; }
    else if(num_processors >= 16) { num_threads = 16; }
    else { num_threads = num_processors; }
	printf("Detected %ld logical processor(s). Using %i thread(s) for parallelized implementation.\n", num_processors, num_threads);

	char *str1, *str2;
	size_t len;

	char option;
	printf("Run edit distance calculation with two random strings of len=%lu (1) OR two custom strings (2)?", n);
	int res = scanf(" %c", &option);
	printf("\n");

	// User chose to use custom strings
	if (res == 1 && option == '2') {
		// Flush
		int c;
		while ((c = getchar()) != '\n' && c != EOF);

		// getline() capacity variables
		size_t cap1 = 0;
		size_t cap2 = 0;

		printf("Please enter the first string (press ENTER to terminate): ");
		ssize_t len1 = getline(&str1, &cap1, stdin);

		printf("Please enter the second string (press ENTER to terminate): ");
		ssize_t len2 = getline(&str2, &cap2, stdin);

		if (len1 == -1 || len2 == -1) {
			fprintf(stderr, "Error reading input.\n");
			return 1;
		}
		if (len1 > 0 && str1[len1 - 1] == '\n') str1[--len1] = '\0';
		if (len2 > 0 && str2[len2 - 1] == '\n') str2[--len2] = '\0';

		if (len1 != len2) {
			printf("Strings must be of equal length. Truncating the longer string to match the length of the shorter string.\n");
			if (len1 > len2) {
				str1[strlen(str2)] = '\0';
				len1 = len2;
			}
			else {
				str2[strlen(str1)] = '\0';
				len2 = len1;
			}
		}
		len = (size_t)len1;
		printf("Custom strings loaded (length: %zu).\n\n", len);
	}

	// User chose random strings OR unrecognized input.
	else{
		if (!(res == 1 && option == '2')) {
			printf("Could not recognize input. Defaulting to calculation with two random strings.\n");
		}
		len = n;
		/*
		* Generate the two random strings of size n.
		* Potentially generating predictable sequence of values due to time(NULL) seeding, but this is not an issue
		* for a non-critical application such as this.
 		*/
		srand(time(NULL)); // NOLINT
		str1 = generate_random_str(len);
		str2 = generate_random_str(len);

		/*
		* Ensure str1 and str2 were initialized
		*/
		if(str1 == NULL || str2 == NULL){
			free(str1);
			free(str2);
			printf("Unable to initialize str1 and/or str2, exiting program\n");
			return 1;
		}
		printf("Random strings (A-Z) successfully generated.\n");
		printf("\n");
	}

	// Run with initialized str1 and str2.
	printf("Running implementations...\n");

	/*
	* Gather runtimes
	*/
	printf("Running naive_edit_distance() as a baseline...\n");
	fflush(stdout);

	double start1 = returnCurrentTime();
	int edit_distance1 = naive_edit_distance(str1, str2, len);
	double end1 = returnCurrentTime();
	printf("naive_edit_distance() calculation finished. Running tiled_edit_distance()...\n");
	fflush(stdout);

	double start2 = returnCurrentTime();
	int edit_distance2 = tiled_edit_distance(str1, str2, len);
	double end2 = returnCurrentTime();
	printf("tiled_edit_distance() calculation finished. Running parallelized_edit_distance()...\n");
	fflush(stdout);

	double start3 = returnCurrentTime();
	int edit_distance3 = parallelized_edit_distance(str1, str2, len, num_threads);
	double end3 = returnCurrentTime();
	printf("parallelized_edit_distance() calculation finished. Running avx2_edit_distance()...\n");
	fflush(stdout);

	double start4 = returnCurrentTime();
	int edit_distance4 = avx2_edit_distance(str1, str2, len, num_threads);
	double end4 = returnCurrentTime();
	printf("avx2_edit_distance() calculation finished.\n");
	fflush(stdout);

	/*
	* Print the results
	*/
	printf("\n");
	printf("naive_edit_distance() result: %i\n", edit_distance1);
	printf("tiled_edit_distance() result: %i\n", edit_distance2);
	printf("parallelized_edit_distance() result: %i\n", edit_distance3);
	printf("avx2_edit_distance() result: %i\n", edit_distance4);
	if(edit_distance1 == edit_distance2 &&
		edit_distance2 == edit_distance3 &&
		edit_distance3 == edit_distance4){
		printf("Results are the same!\n");
		}
	else{
		printf("Results are not the same!\n");
	}

	/*
	* Print the runtimes
	*/
	printf("\n");
	printf("naive_edit_distance() runtime: %.3f s\n", (end1-start1)/1000);
	printf("tiled_edit_distance() runtime: %.3f s\n", (end2-start2)/1000);
	printf("parallelized_edit_distance() runtime: %.3f s\n", (end3-start3)/1000);
	printf("avx2_edit_distance() runtime: %.3f s\n", (end4-start4)/1000);
	printf("\n");

	/*
	* Cleanup
	*/
	free(str1);
	free(str2);
	return 0;
}