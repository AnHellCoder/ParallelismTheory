#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>
#include <cublas_v2.h>

int main(int argc, char** argv){
	clock_t begin = clock();

	cublasHandle_t handler;
	cublasStatus_t status;

	status = cublasCreate(&handler);
	if(status == CUBLAS_STATUS_SUCCESS){
		printf("Success!\n");
	}
	else{
		printf("ERROR!");
		exit(0);
	}

	float err = 1.0, exact = 0.000001;
	int size = atoi(argv[1]), itermax = atoi(argv[2]), iter = 0;

	float* arrprev, *arrnew, *arrerr;

	arrprev = malloc(sizeof(float) * (size * size));
	arrnew = malloc(sizeof(float) * (size * size));
	arrerr = malloc(sizeof(float) * (size * size));
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			arrprev[i * size + j] = 0;
		}
	}

	int k = size - 1;
	arrprev[0] = 10;
	arrprev[k] = 20;
	arrprev[k * size] = 20;
	arrprev[k * size + k] = 30;
	float step = (float)10/size;
	for(int i = 1; i < k; i++){
		arrprev[i] = arrprev[i - 1] + step;
		arrprev[i * size] = arrprev[(i - 1) * size] + step;
		arrprev[k * size + i] = arrprev[k * size + (i - 1)] + step;
		arrprev[i * size + k] = arrprev[(i - 1) * size + k] + step;
	}

	//#pragma acc data copyin(arrprev[:size], arrnew[:size], arrerr[:size])
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			arrnew[i * size + j] = arrprev[i * size + j];
		}
	}

	#pragma acc data copy(err) copyin(arrprev[:size], arrnew[:size], arrerr[:size])
	{
	while(iter < itermax && err > exact){
		iter++;
		double alpha = -1.0;
		int index = 0;

		#pragma acc data present(arrprev, arrnew)
		//#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256)
		#pragma acc parallel loop gang worker num_workers(4) vector_length(128)
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				int n = i * size + j;
				int w = (i - 1) * size + j;
				int x = (i + 1) * size + j;
				int y = i * size + (j - 1);
				int z = i * size + (j + 1);
				arrnew[n] = 0.25 * (arrprev[w] + arrprev[x] + arrprev[y] + arrprev[z]);
			}
		}

		//status = cublasDcopy(handler, size, arrnew, 1, arrerr, 1);

		//#pragma acc data present(arrprev, arrerr)

		if(iter % 100 == 0){
			#pragma acc data present(arrprev, arrnew, arrerr)
			#pragma acc host_data use_device(arrprev, arrnew, arrerr)
			{
			status = cublasDcopy(handler, size, arrnew, 1, arrerr, 1);
			if(status != CUBLAS_STATUS_SUCCESS){
				exit(EXIT_FAILURE);
			}

			status = cublasDaxpy(handler, size, &alpha, arrprev, 1, arrerr, 1);
			if(status != CUBLAS_STATUS_SUCCESS){
				exit(EXIT_FAILURE);
			}

			status = cublasIdamax(handler, size, arrerr, 1, &index);
			if(status != CUBLAS_STATUS_SUCCESS){
				exit(EXIT_FAILURE);
			}
			}

			#pragma acc update host(arrerr[index - 1])
			err = abs(arrerr[index - 1]);
		}
			

		float** temp = arrprev;
		arrprev = arrnew;
		arrnew = temp;

		if(iter == 1 || iter % 1000 == 0){
			printf("On %d iteration error equals %lf\n", iter, err);
			clock_t mid = clock();

			double te_mid = (double)(mid - begin)/CLOCKS_PER_SEC;
			printf("Time elapsed: %lf\n", te_mid);
		}
	}
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