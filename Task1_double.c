#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fill(double* arr){
	double j = (2 * 3.14) / 10000000;
	
	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++){
		arr[i] = sin(j);

		j += (2 * 3.14) / 10000000;
	}
	}
}

double calculate(double* arr){
	double sum = 0;

	#pragma acc kernels
	{
	for(int i = 0; i < 10000000; i++) sum += arr[i];
	}

	return sum;
}

int main(){
	double* arr;
	double sum;

	arr = malloc(sizeof(double) * 10000000);

	fill(arr);

	sum = calculate(arr);

	printf("%lf\n", sum);

	return 0;
}
