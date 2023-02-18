#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fill(float* arr){
	double j = (2 * 3.14) / 10000000;

	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++){
		arr[i] = sin(j);

		j += (2 * 3.14) / 10000000;
	}
	}
}

double calculate(float* arr){
	double sum = 0;

	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++) sum += arr[i];
	}

	return sum;
}

int main(){
	float* arr;
	double sum;
	
	arr = malloc(sizeof(float) * 10000000);

	fill(arr);

	sum = calculate(arr);

	printf("%lf", sum);

	return 0;
}
