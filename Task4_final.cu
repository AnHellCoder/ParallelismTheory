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

//Compute a new element of an array
__global__ void compute(double *arrprev, double *arrnew, int size){

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((j == 0) || (i == 0) || (i >= size - 1) || (j >= size - 1))
                return; // Don't update borders

    arrnew[i * size + j] = 0.25 * (arrprev[i * size + (j - 1)] + arrprev[(i - 1) * size + j] + arrprev[(i + 1) * size + j] + arrprev[i * size + (j + 1)]);
}

//Initialize an array
__global__ void init(double* arr, int size){
        size_t i = threadIdx.x;

        if (i >= size) return; //Error processing

        arr[i] = 10.0 + i * 10.0 / (size - 1);
        arr[i * size] = 10.0 + i * 10.0 / (size - 1);
        arr[size - 1 + i * size] = 20.0 + i * 10.0 / (size - 1);
        arr[size * (size - 1) + i] = 20.0 + i * 10.0 / (size - 1);
}

//Calculate the difference between two arrays
__global__ void loss_calculation(double* arrprev, double* arrnew, double* arrloss, int size) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if(j >= size || i >= size) return; //Error processing

        arrloss[i * size + j] = arrprev[i * size + j] - arrnew[i * size + j];
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

        //Initialize the main attributes
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaGraph_t graph;
        cudaGraphExec_t graph_instance;

        dim3 blocks = dim3(size / 32, size / 32);
        dim3 threads = dim3(32, 32);

        double *arrprev, *arrnew, *arrloss, *cudaLoss, *temp_storage = NULL;
        size_t ts_bytes = 0;

        cudaMalloc(&arrprev, sizeof(double) * (size * size));
        cudaMalloc(&arrnew, sizeof(double) * (size * size));
        cudaMalloc(&arrloss, sizeof(double) * (size * size));
        cudaMalloc(&cudaLoss, sizeof(double));

        init<<<1, size>>>(arrprev, size);
        printArr<<<1, 1>>>(arrprev, size);

        cudaDeviceSynchronize();

        exit(0);

        cudaMemcpy(arrnew, arrprev, sizeof(double) * (size * size), cudaMemcpyDeviceToDevice);

        cub::DeviceReduce::Max(temp_storage, ts_bytes, arrnew, cudaLoss, (size * size), stream);
        cudaMalloc(&temp_storage, ts_bytes);
        ////////////////////////////////////////////////////////////Graph initialization
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        for (size_t i = 0; i < 100; i++) {
                compute<<<blocks, threads, 0, stream>>>(arrprev, arrnew, size);
                swap(arrprev, arrnew);
        }
        loss_calculation<<<blocks, threads, 0, stream>>>(arrprev, arrnew, arrloss, size);
        cub::DeviceReduce::Max(temp_storage, ts_bytes, arrloss, cudaLoss, (size * size), stream);
        init<<<1, size, 0, stream>>>(arrnew, size);

        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);
        /////////////////////////////////////////////////////////////End of graph

        //Main loop
        while(iter < lim && loss > acc) {
                cudaGraphLaunch(graph_instance, stream);
                cudaMemcpyAsync(&loss, cudaLoss, sizeof(double), cudaMemcpyDeviceToHost);
                iter += 100;

                cout << "On " << iter << " iteration loss equals: " << loss << endl;

                clock_t mid = clock();

                cout << "Time elapsed: " << (double)(mid - begin)/CLOCKS_PER_SEC << endl;
        }

        //Results
        clock_t end = clock();

        cout << "After " << iter << " iterations loss descended to " << loss << endl;
        cout << "Time elapsed: " << (double)(end - begin)/CLOCKS_PER_SEC << endl;

        cudaFree(arrprev);
        cudaFree(arrnew);
        cudaFree(arrloss);
        cudaFree(cudaLoss);
        cudaFree(temp_storage);

        return 0;
}
