main: main.c
	gcc -g -Wall -march=native -O3 -Isrc -o main main.c naive_edit_distance.c tiled_edit_distance.c parallelized_edit_distance.c avx2_edit_distance.c -lpthread
test: unused/test_edit_distance.c
	gcc -Wall -O3 -march=native -o -Isrc main_test test_edit_distance.c naive_edit_distance.c tiled_edit_distance.c parallelized_edit_distance.c avx2_edit_distance.c -lpthread
.PHONY: clean
clean:
	rm -f main *.o