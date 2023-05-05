#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;

void print_error(){
	cout << "Arguments were not parsed correctly!" << endl;
	cout << "Print --help to get help" << endl;
}

void print_help(){
	cout << "How to send args through cmd:" << endl;
	cout << "--accuracy <double> --size <int> --limit <int>" << endl;
}

__global__ void init(double* arr, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i == size) return;

	double step = ((double)10/size) * i;

	int k = size - 1;
	arr[i * size] = 10 + step;
	arr[i] = 10 + step;
	arr[k * size + i] = 20 + step;
	arr[i * size + k] = 20 + step;
}

__global__ void compute(double* arrnew, double* arrprev, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(blockIdx.x == 0 || blockIdx.x == size - 1) return;
	if(threadIdx.x == 0 || threadIdx.x == size - 1) return;

	arrnew[i] = 0.25 * (arrprev[i - 1] + arrprev[i + 1] + arrprev[i - size] + arrprev[i + size]);
}

__host__ double loss_recalculate(double loss, double* arrnew, double* arrprev){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	loss = fmax(loss, arrnew[i] - arrprev[i]);

	return loss;
}

int main(int argc, char* argv[]){
	double acc, loss = 1.0;
	int iter, lim, size;

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

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	double* arrprev;
	double* arrnew;

	cudaMalloc(&arrprev, sizeof(double) * (size * size));
	cudaMalloc(&arrnew, sizeof(double) * (size * size));

	init<<<size, size>>>(arrprev, size);
	init<<<size, size>>>(arrnew, size);

	compute<<<size, size>>>(arrnew, arrprev, size);

	loss_recalculate<<<size, size>>>(loss, arrnew, arrprev);

	cout << loss << endl;

	return 0;
}
