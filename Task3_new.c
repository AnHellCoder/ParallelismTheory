#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>
#include <cublas_v2.h>

void printez(float* arr, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("%lf ", arr[i * size + j]);
		}
	}
}

int main(int argc, char** argv){
	clock_t begin = clock();

	//Handle and status
	cublasHandle_t handler;
	cublasStatus_t status;

	//Initialize a context
	status = cublasCreate(&handler);
	if(status == CUBLAS_STATUS_SUCCESS){
		printf("Success!\n");
	}
	else{
		printf("ERROR!");
		exit(0);
	}

	//initialization
	double err = 1.0, exact = 0.000001;
	int size = atoi(argv[1]), itermax = atoi(argv[2]), iter = 0;

	double* arrprev, *arrnew, *arrerr;

	arrprev = malloc(sizeof(double) * (size * size));
	arrnew = malloc(sizeof(double) * (size * size));
	arrerr = malloc(sizeof(double) * (size * size));
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			arrprev[i * size + j] = 0;
		}
	}

	//initial fill of the array
	int k = size - 1;
	arrprev[0] = 10;
	arrprev[k] = 20;
	arrprev[k * size] = 20;
	arrprev[k * size + k] = 30;
	double step = (double)10/size;
	for(int i = 1; i < k; i++){
		arrprev[i] = arrprev[i - 1] + step;
		arrprev[i * size] = arrprev[(i - 1) * size] + step;
		arrprev[k * size + i] = arrprev[k * size + (i - 1)] + step;
		arrprev[i * size + k] = arrprev[(i - 1) * size + k] + step;
	}

	//Copy old array to new array
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			arrnew[i * size + j] = arrprev[i * size + j];
		}
	}

	//Start computing
	#pragma acc data copy(err) copyin(arrprev[:size*size], arrnew[:size*size], arrerr[:size*size]) //Copy data to GPU
	{
	while(iter < itermax && err > exact){
		iter++;
		double alpha = -1.0;
		int index = 0;

		#pragma acc data present(arrprev, arrnew) //Update pointers
		#pragma acc parallel loop collapse(2) gang worker num_workers(4) vector vector_length(128) //Collapse two loops into single
		//Calculating new cell
		for(int i = 1; i < size - 1; i++){
			for(int j = 1; j < size - 1; j++){
				int n = i * size + j;
				int w = (i - 1) * size + j;
				int x = (i + 1) * size + j;
				int y = i * size + (j - 1);
				int z = i * size + (j + 1);
				arrnew[n] = 0.25 * (arrprev[w] + arrprev[x] + arrprev[y] + arrprev[z]);
			}
		}

		//Calculate the error every 100 iterations
		if(iter % 100 == 0){
			#pragma acc data present(arrprev, arrnew, arrerr)
			#pragma acc host_data use_device(arrprev, arrnew, arrerr)
			{
			//Copy new array to error array
			status = cublasDcopy(handler, size * size, arrnew, 1, arrerr, 1);
			if(status != CUBLAS_STATUS_SUCCESS){
				printf("COPY ERROR!");
				exit(EXIT_FAILURE);
			}

			//Calculate the error, substracting old array array from copy of new array
			status = cublasDaxpy(handler, size * size, &alpha, arrprev, 1, arrerr, 1);
			if(status != CUBLAS_STATUS_SUCCESS){
				printf("AXPY ERROR!");
				exit(EXIT_FAILURE);
			}

			//Find maximum from the result of previous function
			status = cublasIdamax(handler, size * size, arrerr, 1, &index);
			if(status != CUBLAS_STATUS_SUCCESS){
				printf("MAX ERROR!");
				exit(EXIT_FAILURE);
			}
			}

			#pragma acc update host(arrerr[index - 1])
			err = arrerr[index - 1];
		}
			

		//Array swap
		double* temp = arrprev;
		arrprev = arrnew;
		arrnew = temp;

		//Print the results every 1000 iterations
		if(iter == 1 || iter % 1000 == 0){
			printf("On %d iteration error equals %lf\n", iter, err);
			clock_t mid = clock();

			double te_mid = (double)(mid - begin)/CLOCKS_PER_SEC;
			printf("Time elapsed: %lf\n", te_mid);
		}
	}

	//Results
	printf("On %d iteration error descended to %lf\n", iter, err);
	}

	free(arrprev);
	free(arrnew);
	free(arrerr);
	cublasDestroy(handler);

	clock_t end = clock();
	double te = (double)(end - begin)/CLOCKS_PER_SEC;
	printf("Time elapsed: %lf\n", te);

	return 0;
}
