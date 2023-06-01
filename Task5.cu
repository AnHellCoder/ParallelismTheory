#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>

#define DEVICE_MALLOC(type,arg,size) type* arg; cudaMalloc((void**)&arg, sizeof(type) * size);

using namespace std;

__global__ void compute(double* arrprev, double* arrnew, int size){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x==0 || y==0 || x==size - 1 || y==size - 1) return;

        arrnew[y * size + x] = 0.25 * (arrprev[y * size + (x - 1)] + arrprev[y * size + (x + 1)] + arrprev[(y - 1) * size + x] + arrprev[(y + 1) * size + x]);
}

__global__ void loss_calculation(double* arrprev, double* arrnew, double* arrloss){
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        arrloss[i] = arrnew[i] - arrprev[i];
}

int main(int argc,char *argv[]){
        clock_t begin = clock();

        //MPI initialization
        int rank, group_size, status;
        MPI_Request request;
        MPI_Init(&argc,&argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &group_size);

        //CUDA check device count
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount<group_size) throw runtime_error("Too many MPI threads");

        cudaSetDevice(rank);
        cout << "Your rank is " << rank << ", group_size is " << group_size << endl;

        //Set p2p access
        if (rank!=0) cudaDeviceEnablePeerAccess(rank-1,0);
        if (rank!=group_size-1) cudaDeviceEnablePeerAccess(rank+1,0);


        //Initialization
        double acc = 0.000001, loss = 1.0, loss_buff;
        int size = 1024, lim = 1000000, iter = 0;

        if(argc > 1){
                if(argc == 2 && string(argv[1]) == "--help"){
                        cout << "How to send args through cmd:" << endl;
                        cout << "--accuracy <double> --size <int> --limit <int>" << endl;
                        exit(0);
                }
                if(string(argv[1]) == "--accuracy") acc = atof(argv[2]);
                else{
                        throw runtime_error("Invalid argument sent. Send '--help' argument to get help");
                }
                if(string(argv[3]) == "--size") size = atoi(argv[4]);
                else{
                        throw runtime_error("Invalid argument sent. Send '--help' argument to get help");
                        exit(0);
                }
                if(string(argv[5]) == "--limit") lim = atoi(argv[6]);
                else{
                        throw runtime_error("Invalid argument sent. Send '--help' argument to get help");
                        exit(0);
                }
        }

        int start = size * rank / group_size-1;
        int end = size * (rank+1) / group_size+1;

        if (rank==0) start+=1;
        if (rank==group_size-1) end-=1;

        int size_per_gpu = end-start;
        if (group_size==1) size_per_gpu=size;

        DEVICE_MALLOC(double,arrprev,(size * size))
        DEVICE_MALLOC(double,arrnew,(size * size))
        DEVICE_MALLOC(double,arrloss,(size * size))
        DEVICE_MALLOC(double,cudaLoss,1)

        //Temporary array initialization
        double* arr = new double[(size * size)];
        memset(arr, 0, (size * size)*sizeof(double));

        //Threads and blocks initialization
        int threads_x = size < 1024 ? size : 1024;
        int blocks_y = size_per_gpu;
        int blocks_x = size/threads_x;

        dim3 blocks(blocks_x, blocks_y);
        dim3 threads(threads_x,1);

        //Temporary array initialization (fill)
        double step = (double)10/(size-1);
        int k = size - 1;
        arr[0] = 10;
        arr[size - 1] = 20;
        arr[(size - 1) * size] = 20;
        arr[(size - 1) * size + (size - 1)] = 30;

        for(int i = 1; i < size; i++){
                arr[i] = arr[i - 1] + step;
                arr[i * size + k] = arr[(i - 1) * size + k] + step;
                arr[k * size + i] = arr[k * size + (i - 1)] + step;
                arr[i * size] = arr[(i - 1) * size] + step;
        }

        //Copy to GPU
        cudaMemcpy(arrprev,arr, sizeof(double)*(size * size), cudaMemcpyHostToDevice);
        cudaMemcpy(arrnew,arr, sizeof(double)*(size * size), cudaMemcpyHostToDevice);

        //CUB initialization
        void *temp_storage = NULL;
        size_t ts_bytes = 0;

        cub::DeviceReduce::Max(temp_storage, ts_bytes, arrloss, cudaLoss, (size * size));
        cudaMalloc(&temp_storage, ts_bytes);

        //Main loop
        while(iter < lim && loss > acc){
                iter++;

                compute<<<blocks,threads>>>(arrprev, arrnew, size);

                //Calculate loss every 100 iterations
                if (iter % 100 == 0){
                        loss_calculation<<<size,size>>>(arrprev, arrnew, arrloss);

                        cub::DeviceReduce::Max(temp_storage, ts_bytes, arrloss, cudaLoss, (size * size));
                        cudaMemcpy(&loss,cudaLoss, sizeof(double), cudaMemcpyDeviceToHost);

                        //Sending loss to all processes
                        status = MPI_Allreduce(&loss,&loss_buff,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                        if(status != MPI_SUCCESS){
                                cout << "REDUCE ERROR!" << endl;
                                return 1;
                        }

                        //Intermediate results
                        clock_t mid = clock();
                        cout << "On " << iter << " iteration loss equals: " << loss << endl;
                        cout << "Time elapsed: " << (double)(mid - begin)/CLOCKS_PER_SEC << endl;
                        cout << "Computed on " << rank << " device" << endl;
               }

               //Exchanging matrix rows between ranks
               if (rank!=group_size-1){
                       MPI_Isend(&arrnew[(size_per_gpu-2)*size+1],size-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&request);
                       MPI_Recv(&arrnew[(size_per_gpu-1)*size+1],size-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                       //status = MPI_Sendrecv(&arrnew[(size_per_gpu-2)*size+1],size-2,MPI_DOUBLE,rank+1,0,&arrnew[(size_per_gpu-1)*size+1],size-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
               }
               if (rank!=0){
                       MPI_Isend(&arrnew[size+1],size-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,&request);
                       MPI_Recv(&arrnew[1],size-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                       //status = MPI_Sendrecv(&arrnew[size+1],size-2,MPI_DOUBLE,rank-1,0,&arrnew[1],size-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
               }

               swap(arrprev,arrnew);
        }

        //Results
        if(rank != 0) MPI_Send(&loss,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
        else{
                double loss_import;
                for(int i = 1; i < group_size; i++){
                        MPI_Recv(&loss_import,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        loss = max(loss, loss_import);
                }
                clock_t te = clock();
                cout << "On " << iter << " iteration loss descended to " << loss << endl;
                cout << "Time elapsed: " << (double)(te - begin)/CLOCKS_PER_SEC << endl;
        }

        cudaFree(arrprev);
        cudaFree(arrnew);
        cudaFree(arrloss);
        cudaFree(cudaLoss);
        delete[] arr;

        MPI_Finalize();

        return 0;
}