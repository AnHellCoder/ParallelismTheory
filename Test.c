#define _USE_MATH_DEFINES
#define N pow(10, 7)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>

int main(){
	double* arr;
	double j = (2 * M_PI) / N, sum = 0;

	arr = malloc(sizeof(double) * N);

	for(int i = 0; i < N; i++){
		arr[i] = j;

		j += (2 * M_PI) / N;
	}

	for(int i = 0; i < N; i++) sum += arr[i];

	printf("%.074f\n", sum);

	return 0;
}
