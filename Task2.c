#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include "nvToolsExt.h"
//#include <nvtx3/nvToolsExt.h>

int iter;

float searchMax(float** arrcur, float** arrprev, int size) {
	float** arrdelta = malloc(sizeof(float*) * size);
	float max = 0;

	#pragma acc loop vector
	for (int i = 0; i < size; i++) {
		arrdelta[i] = malloc(sizeof(float) * size);
	}
	iter++;

	#pragma acc loop gang
	for (int i = 0; i < size; i++) {
		#pragma acc loop vector
		for (int j = 0; j < size; j++) {
			arrdelta[i][j] = fabs(arrcur[i][j] - arrprev[i][j]);
			if (arrdelta[i][j] > max) max = arrdelta[i][j];
		}
	}

	return max;
}

void arrayCompute(int size, float** arr){
	int k = size - 1;
	arr[0][0] = 10, arr[0][k] = 20;
	arr[k][0] = 30, arr[k][k] = 20;

	#pragma acc loop vector
	for (int i = 1; i < k; i++) {
		arr[0][i] = fabs(arr[0][0] - arr[0][k]) / k;
		arr[k][i] = fabs(arr[k][0] - arr[k][k]) / k;
		arr[i][0] = fabs(arr[0][0] - arr[k][0]) / k;
		arr[i][k] = fabs(arr[0][k] - arr[k][k]) / k;
	}

	int i = 1, j = 1;

	while (i < size - 1) {
		int sq = size - 1;
		arr[i][j] = (arr[i][0] + arr[i][sq] + arr[0][j] + arr[sq][j]) / 4;

		if (j == size - 1) {
			i++;
			j = 1;
		}
		else j++;
	}
	iter++;
}

int main(int argc, char* argv[]){
//	nvtxRangePushA("GPU Time");

	clock_t begin = clock();

	float exact = atof(argv[1]), err = INT_MAX;
	int size = atoi(argv[2]), iterlim = atoi(argv[3]);

	while(iter < iterlim && err > exact){
		float** arrcur, **arrprev;

		arrcur = malloc(sizeof(float*) * size);
		arrprev = malloc(sizeof(float*) * size);
	
		#pragma acc loop vector
		for(int i = 0; i < size; i++){
			arrcur[i] = malloc(sizeof(float) * size);
			arrprev[i] = malloc(sizeof(float) * size);
		}

		arrayCompute(size, arrcur);
		arrayCompute(size, arrprev);

		float maxError = searchMax(arrcur, arrprev, size);

		if(maxError < err) err = maxError;
		
		#pragma acc loop vector
		for(int i = 0; i < size; i++){
			free(arrcur[i]);
			free(arrprev[i]);
		}

		free(arrcur);
		free(arrprev);
	}

	printf("For %d iterations maximal error value is: %f\n", iter, err);

	clock_t end = clock();

	double time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("%lf", time_elapsed);

//	nvtxRangePop();

	return 0;
}
