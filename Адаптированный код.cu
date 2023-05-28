#include <iostream>
#include <mpi.h>
#include <cmath>
#include <string>
#include <sstream>
#include <exception>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>

#define CREATE_DEVICE_ARR(type,arg,size) type* arg; cudaMalloc((void**)&arg, sizeof(type) * size);
#define CUDACHECK(name) if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(name);

template <typename Type>
std::istream& operator>>(std::istream& i, const Type& arg) { return i; }

template <typename Type>
void argpars(Type& arg, std::string& str){
        std::stringstream buff;
        buff << str;
        buff >> arg;
        std::string buff2;
        buff2 = str;
        str.clear();
        std::getline(buff, str);
        if (str == buff2) throw std::runtime_error("Not a valid argument");
}

__global__ void interpolate(double* A,double* Anew,unsigned int size_x,unsigned int size_y){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x==0 || y==0 || x==size_x-1 || y==size_y-1) return;

        Anew[y*size_x+x] = 0.25 * (A[y*size_x+x-1] + A[y*size_x+x+1] + A[(y-1)*size_x+x] + A[(y+1)*size_x+x]);
}

__global__ void difference(double* A,double* B){
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        A[i] -= B[i];
}

int NOD(int a, int b){
        while(a > 0 && b > 0){
                if(a > b) a %= b;
                else b %= a;
        }

        return a + b;
}

int main(int argc,char *argv[]){
        //Init default values
        int rank, group_size;
        MPI_Request request;
        MPI_Init(&argc,&argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &group_size);

        //CUDA check device count
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount<group_size) throw std::runtime_error("Too many MPI threads");

        cudaSetDevice(rank);

        //Set p2p access
        if (rank!=0) cudaDeviceEnablePeerAccess(rank-1,0);
        if (rank!=group_size-1) cudaDeviceEnablePeerAccess(rank+1,0);


        //default settings
        double acc = 0.000001;
        int size=1024;
        int lim = 1000000;

        //Reading arguments
        for (int i =1;i<argc-1;i+=2){
                std::string argument(argv[i]);
                std::string value(argv[i+1]);

                if (argument=="--accuracy") argpars(acc,value);
                else if (argument=="--size") argpars(size,value);
                else if (argument=="--limit") argpars(lim,value);
        }


        //Init net and buffer
        int start = size * rank / group_size-1;
        int end = size * (rank+1) / group_size+1;

        if (rank==0) start+=1;
        if (rank==group_size-1) end-=1;

        int size_per_gpu = end-start;

        if (group_size==1) size_per_gpu=size;

        int net_size = size_per_gpu*size;

        double* net_cpu = new double[net_size];
        memset(net_cpu,0,net_size*sizeof(double));

        CREATE_DEVICE_ARR(double,buff,net_size)
        CREATE_DEVICE_ARR(double,net,net_size)
        CREATE_DEVICE_ARR(double,net_buff,net_size)
        CREATE_DEVICE_ARR(double,d_out,1)


        //Corners
        double lu = 10;
        double ru = 20;
        double ld = 20;
        double rd = 30;

        //Threads and blocks init
        unsigned int threads_x=NOD(size,1024);
        unsigned int blocks_y = size_per_gpu;
        unsigned int blocks_x = size/threads_x;

        dim3 dim_for_interpolate(threads_x,1);
        dim3 block_for_interpolate(blocks_x,blocks_y);

        //Fill default values
        double step = (double)10/size;
        int k = size - 1;
        net_cpu[0] = lu;
        net_cpu[size - 1] = ru;
        net_cpu[(size - 1) * size] = ld;
        net_cpu[(size - 1) * size + (size - 1)] = rd;

        for(int i = 1; i < size; i++){
                net_cpu[i] = net_cpu[i - 1] + step;
                net_cpu[i * size + k] = net_cpu[(i - 1) * size + k] + step;
                net_cpu[k * size + i] = net_cpu[k * size + (i - 1)] + step;
                net_cpu[i * size] = net_cpu[(i - 1) * size] + step;
        }

        cudaMemcpy(net,net_cpu, sizeof(double)*net_size, cudaMemcpyHostToDevice);
        cudaMemcpy(net_buff,net_cpu, sizeof(double)*net_size, cudaMemcpyHostToDevice);

        //Init cycle values
        int iter = 0;
        double max_acc=1.0,max_acc_buff;

        //Cub init
        void *d_temp_storage = NULL;
        size_t  temp_storage_bytes = 0;

        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);


        //Start solving
        //for (iter = 0;iter <lim;iter++){
        while(iter < lim && max_acc > acc){
                //Set the new array
                iter++;

                interpolate<<<block_for_interpolate,dim_for_interpolate>>>(net,net_buff,size,size_per_gpu);
                CUDACHECK("end");

                //Doing reduction to find max
                if (iter % 100 == 0){
                        cudaMemcpy(buff,net_buff, sizeof(double)*net_size, cudaMemcpyDeviceToDevice);
                        difference<<<blocks_x*blocks_y,threads_x>>>(buff,net);

                        //Finding max accuracy
                        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
                        cudaMemcpy(&max_acc,d_out, sizeof(double), cudaMemcpyDeviceToHost);
                        max_acc = std::abs(max_acc);

                        //Sending max accuracy to all process
                        MPI_Allreduce(&max_acc,&max_acc_buff,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

                        std::cout << "On " << iter << " iteration loss equals: " << max_acc << std::endl;
               }

                //Exchanging matrix rows between ranks
                //This send penultimate and second rows
                //and get last and fisrt rows
                if (rank!=group_size-1){
                        MPI_Isend(&net_buff[(size_per_gpu-2)*size+1],size-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&request);
                        MPI_Recv(&net_buff[(size_per_gpu-1)*size+1],size-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
                if (rank!=0){
                        MPI_Isend(&net_buff[size+1],size-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,&request);
                        MPI_Recv(&net_buff[1],size-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }

                std::swap(net,net_buff);
        }
        CUDACHECK("end");

        //Getting results to first process and printing it
        if(rank!=0) MPI_Send(&max_acc,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
        else{
                double max_acc_buff;
                for (int i=1;i<group_size;i++){
                        MPI_Recv(&max_acc_buff,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        max_acc = std::max(max_acc,max_acc_buff);
                }
                std::cout<<"Iteration count: "<<iter<<"\n";
                std::cout<<"Accuracy: "<<max_acc<<"\n";
        }

        //Finishing program
        cudaFree(net);
        cudaFree(net_buff);
        cudaFree(buff);
        cudaFree(d_out);
        delete[] net_cpu;
        MPI_Finalize();

        return 0;
}