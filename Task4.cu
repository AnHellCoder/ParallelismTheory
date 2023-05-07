#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;

//Print when incorrect args sent
void print_error(){
	cout << "Arguments were not parsed correctly!" << endl;
	cout << "Print --help to get help" << endl;
}

//Print when arg '--help' sent
void print_help(){
	cout << "How to send args through cmd:" << endl;
	cout << "--accuracy <double> --size <int> --limit <int>" << endl;
}

//Array initialization
__global__ void init(double* arr, int size){
	int k = size - 1;
	double step = (double)10/size;

	arr[0] = 10;
	arr[k] = 20;
	arr[k * size] = 20;
	arr[k * size + k] = 30;
	for(int i = 1; i < k; i++){
		arr[i] = arr[i - 1] + step;
		arr[k * size + i] = arr[k * size + (i - 1)] + step;
		arr[i * size] = arr[(i - 1) * size] + step;
		arr[i * size + k] = arr[(i - 1) * size + k] + step;
	}
}

//compute new array
__global__ void compute(double* arrnew, double* arrprev, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(blockIdx.x == 0 || blockIdx.x == size - 1) return;
	if(threadIdx.x == 0 || threadIdx.x == size - 1) return;

	arrnew[i] = 0.25 * (arrprev[i - 1] + arrprev[i + 1] + arrprev[i - size] + arrprev[i + size]);
}

//calculate loss
__global__ void loss_calculate(double* arrnew, double* arrprev, double* arrloss){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	arrloss[i] = arrnew[i] - arrprev[i];
}

//print array on GPU
__global__ void printArr(double* arr, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("%lf ", arr[i * size + j]);
		}
		printf("\n");
	}
}

int main(int argc, char* argv[]){
	//Start execution
	clock_t begin = clock();

	double acc, loss = 1.0;
	int iter = 0, lim, size;

	//Argument parsing
	string* args = new string[argc];
	for(int i = 0; i < argc; i++) args[i] = argv[i];

	if(argc == 2 && args[1] == "--help"){
		print_help();
		exit(0);
	}

	if(args[1] == "--accuracy") acc = atof(argv[2]);
	else{
		print_error();
		exit(0);
	}

	if(args[3] == "--size") size = atoi(argv[4]);
	else{
		print_error();
		exit(0);
	}

	if(args[5] == "--limit") lim = atoi(argv[6]);
	else{
		print_error();
		exit(0);
	}
	//End argument parsing

	//Stream initialization
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//Initialization of arrays
	double* arrprev;
	double* arrnew;
	double* arrloss;
	double* temp_storage = NULL;
	double* cudaLoss;

	size_t ts_bytes;

	cudaMalloc(&cudaLoss, sizeof(double));

	cudaMalloc(&arrprev, sizeof(double) * (size * size));
	cudaMalloc(&arrnew, sizeof(double) * (size * size));
	cudaMalloc(&arrloss, sizeof(double) * (size * size));

	//Primary initialization
	init<<<1, 1>>>(arrprev, size);
	init<<<1, 1>>>(arrnew, size);

	//Start computing
	while(loss > acc && iter <= lim){
		iter++;

		compute<<<size, size>>>(arrnew, arrprev, size);

		//Calculate loss every 100 iterations
		if(iter % 100 == 0){
			loss_calculate<<<size, size>>>(arrnew, arrprev, arrloss);

			//Calculate max loss in loss array and read it into one-element array cudaLoss
			cudaMalloc(&cudaLoss, sizeof(double));
			cub::DeviceReduce::Max(temp_storage, ts_bytes, arrloss, cudaLoss, (size * size));
			cudaMalloc(&temp_storage, ts_bytes);
			cub::DeviceReduce::Max(temp_storage, ts_bytes, arrloss, cudaLoss, (size * size));

			//read cudaLoss value into loss variable
			cudaMemcpy(&loss, cudaLoss, sizeof(double), cudaMemcpyDeviceToHost);

			//Print elapsed time, number of passed iterations and actual loss
			clock_t mid = clock();
			double te = (double)(mid - begin)/CLOCKS_PER_SEC;

			cout << "On " << iter << " iteration loss equals: " << loss << endl;
			cout << "Time elapsed: " << te << endl;
		}

		//Array swap
		swap(arrprev, arrnew);
	}
	//End computing

	return 0;
}
