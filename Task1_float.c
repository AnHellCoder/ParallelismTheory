#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fill(float* arr){
	double j = (2 * 3.14) / 10000000;

	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++){
		arr[i] = sinf(j);

		j += (2 * 3.14) / 10000000;
	}
	}
}

float calculate(float* arr){
	float sum = 0;

	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++) sum += arr[i];
	}

	return sum;
}

int main(){
	float* arr;
	float sum;
	
	arr = malloc(sizeof(float) * 10000000);

	fill(arr);

	sum = calculate(arr);

	printf("%lf", sum);

	return 0;
}
