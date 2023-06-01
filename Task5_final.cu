#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

using namespace std;

__global__ void compute(double* arrnew, double* arrprev, int size, int groupSize){
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < groupSize - 1 && j > 0 && j < size - 1) {
        arrnew[i * size + j] = 0.25 * (arrprev[i * size + j - 1] + arrprev[(i - 1) * size + j] + arrprev[(i + 1) * size + j] + arrprev[i * size + j + 1]);
    }
}


__global__ void loss_calculation(double* arrnew, double* arrprev, double* arrloss){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    arrloss[i] = abs(arrprev[i] - arrnew[i]);
}

int main(int argc, char** argv) {
    clock_t begin = clock();

    int rank, total_ranks;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);


    cudaSetDevice(rank);

    if (rank!=0)
        cudaDeviceEnablePeerAccess(rank - 1, 0);
    if (rank!=total_ranks-1)
        cudaDeviceEnablePeerAccess(rank + 1, 0);

    int size = 1024, lim = 1000000, iter_count = 0;
    double acc = 0.000001, loss = 1.0;

    if(argc > 1){
            if(string(argv[1]) == "--accuracy") acc = atof(argv[2]);
            else{
                    cout << "Invalid argument sent. Send '--help' argument to get help" << endl;
                    exit(0);
            }

            if(string(argv[3]) == "--size") size = atoi(argv[4]);
            else{
                    cout << "Invalid argument sent. Send '--help' argument to get help" << endl;
                    exit(0);
            }

            if(string(argv[5]) == "--limit") lim = atoi(argv[6]);
            else{
                    cout << "Invalid argument sent. Send '--help' argument to get help" << endl;
                    exit(0);
            }
    }

    size_t size_per_gpu = size / total_ranks;
    size_t startRow = size_per_gpu * rank;

    double* arr;

    arr = new double[size * size];
    memset(arr, 0, size * size * sizeof(double));

    int k = size - 1;
    arr[0] = 10;
    arr[k * size] = 20;
    arr[k] = 20;
    arr[k * size + k] = 30;

    for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++) cout << arr[i * size + j] << " ";
            cout << endl;
    }

    double step = (double)10/(size-1);
    for (size_t i = 1; i < size - 1; i++) {
        arr[i] = arr[i-1] + step;
        arr[i * size] = arr[(i - 1) * size] + step;
        arr[k * size + i] = arr[k * size + (i - 1)] + step;
        arr[i * size + k] = arr[(i - 1) * size + k] + step;
    }

    for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++) cout << arr[i * size + j] << " ";
            cout << endl;
    }
    return;

    if (rank != 0 && rank != total_ranks - 1) size_per_gpu += 2;
    else size_per_gpu++;

    double* arrnew, *arrprev, *arrloss, *cudaLoss, *temp_storage = NULL;
    cudaMalloc(&arrprev, sizeof(double) * (size * size_per_gpu));
    cudaMalloc(&arrnew, sizeof(double) * (size * size_per_gpu));
    cudaMalloc(&arrloss, sizeof(double) * (size * size_per_gpu));
    cudaMalloc(&cudaLoss, sizeof(double));

    size_t temp_storage_bytes = 0;
    size_t offset = (rank != 0) ? size : 0;

    int threads_x = (size < 1024) ? size : 1024;
    int blocks_y = size_per_gpu;
    int blocks_x = size / threads_x;

    dim3 threads(threads_x, 1);
    dim3 blocks(blocks_x, blocks_y);

    cudaMemcpy(arrprev, arr + (startRow * size) - offset, sizeof(double) * (size * size_per_gpu), cudaMemcpyHostToDevice);
    cudaMemcpy(arrnew, arr + (startRow * size) - offset, sizeof(double) * (size * size_per_gpu), cudaMemcpyHostToDevice);

    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, arrloss, cudaLoss, (size * size_per_gpu));
    cudaMalloc(&temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (iter_count < lim && loss > acc) {
        iter_count += 1;

        compute<<<blocks, threads, 0, stream>>>(arrnew, arrprev, size, size_per_gpu);

        if(iter_count % 100 == 0){
            loss_calculation<<<blocks_x * blocks_y, threads_x, 0, stream>>>(arrnew, arrprev, arrloss);
            cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, arrloss, cudaLoss, (size * size_per_gpu));

            cudaStreamSynchronize(stream);

            MPI_Allreduce(cudaLoss,cudaLoss, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            cudaMemcpyAsync(&loss, cudaLoss, sizeof(double), cudaMemcpyDeviceToHost, stream);

            clock_t mid = clock();
            cout << "On " << iter_count << " iteration loss equals " << loss << endl;
            cout << "Time elapsed: " << (double)(mid - begin)/CLOCKS_PER_SEC << endl;

        }
        cudaStreamSynchronize(stream);

        if (rank != 0){
            MPI_Sendrecv(arrnew + size + 1, size - 2, MPI_DOUBLE, rank - 1, 0, arrnew + 1, size - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank != total_ranks - 1){
            MPI_Sendrecv(arrnew + (size_per_gpu - 2) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0, arrnew + (size_per_gpu - 1) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        swap(arrprev, arrnew);
    }

    if (rank == 0){
        clock_t end = clock();

        cout << "On " << iter_count << " iteration loss descended to " << loss << endl;
        cout << "Time elapsed: " << (double)(end - begin)/CLOCKS_PER_SEC << endl;
    }

    cudaFree(arrprev);
    cudaFree(arrnew);
    cudaFree(temp_storage);
    cudaFree(cudaLoss);
    delete[] arr;

    MPI_Finalize();

    return 0;
}
