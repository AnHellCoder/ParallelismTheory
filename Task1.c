#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(){
	double* arr;

	double j = (2 * 3.14) / 10000000, sum = 0;

	arr = malloc(sizeof(double) * 10000000);

	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++){
		arr[i] = sin(j);

		j += (2 * 3.14) / 10000000;
	}

	for(int i = 0; i < 10000000; i++) sum += arr[i];
	}

	printf("%.074f\n", sum);

	return 0;
}
