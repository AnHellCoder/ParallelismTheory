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

	for(int i = 1; i < size - 1; i++){
		for(int j = 1; j < size - 1; j++) arr[i * size + j] = 0;
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
			printf("%.2f ", arr[i * size + j]);
		}
		printf("\n");
	}
}

int main(int argc, char* argv[]){
	//Initialization
	clock_t begin = clock();

	cudaSetDevice(3);

	double acc, loss = 1.0;
	int iter = 0, lim, size;

	//Arguments preprocessing
	if(argc == 2 && string(argv[1]) == "--help"){
		print_help();
		exit(0);
	}

	if(string(argv[1]) == "--accuracy") acc = atof(argv[2]);
	else{
		print_error();
		exit(0);
	}

	if(string(argv[3]) == "--size") size = atoi(argv[4]);
	else{
		print_error();
		exit(0);
	}

	if(string(argv[5]) == "--limit") lim = atoi(argv[6]);
	else{
		print_error();
		exit(0);
	}

	//Initialization of graph parameters
	cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaGraph_t graph;
        cudaGraphExec_t graph_instance;

	//Array initialization
	double* arrprev, *arrnew, *arrloss, *cudaLoss, *temp_storage = NULL;
	size_t tsbytes = 0;

	cudaMalloc(&arrprev, sizeof(double) * (size * size));
	cudaMalloc(&arrnew, sizeof(double) * (size * size));
	cudaMalloc(&arrloss, sizeof(double) * (size * size));
	cudaMalloc(&cudaLoss, sizeof(double));

	init<<<1, 1>>>(arrprev, size);
	init<<<1, 1>>>(arrnew, size);

	//////////////////////////////////////////////////////////Begin of the graph
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

	//Start of computations
	for(int i = 0; i < 100; i++){
		compute<<<size, size, 0, stream>>>(arrnew, arrprev, size);
		if(i < 99) swap(arrprev, arrnew);
	}
	loss_calculate<<<size, size, 0, stream>>>(arrnew, arrprev, arrloss);
	cub::DeviceReduce::Max(temp_storage, tsbytes, arrloss, cudaLoss, (size * size), stream);

	cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);
	//////////////////////////////////////////////////////////End of the graph

	//Main loop
	while(loss > acc && iter <= lim){
		cudaGraphLaunch(graph_instance, stream);
		cudaMalloc(&temp_storage, tsbytes);
		cub::DeviceReduce::Max(temp_storage, tsbytes, arrloss, cudaLoss, (size * size), stream);

		cudaMemcpy(&loss, cudaLoss, sizeof(double), cudaMemcpyDeviceToHost);

		clock_t mid = clock();
		double te = (double)(mid - begin)/CLOCKS_PER_SEC;

		cout << "On " << iter << " iteration loss equals: " << loss << endl;
		cout << "Time elapsed: " << te << endl;

		iter += 100;
	}

	return 0;
}
