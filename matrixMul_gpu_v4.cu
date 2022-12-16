/*
 * _MATRIXMUL_GPU_CU_
 *
 * 2022 Mert SIDE
 *
 * CS5375 Computer Systems Organization and Architecture 
 * Guest Lecture: GPU Programming
 *
 * Multiplying two matrices on the GPU
 *
 */

 #include <iostream>
 #include <stdio.h>
 #include <stdlib.h>
 
 // CUDA Kernel function to initialize the matrix
 __global__
 void GPUmatmul_init(int N, double *x, double *y, double *ans)
 {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
        x[j*N+i] = 5;
        y[j*N+i] = (i==j?1:0);
        ans[j*N+i] = (double)0.000000000000;
 }


 // CUDA Kernel function to multiple the elements of two arrays
 __global__
 void GPUmatmul(int N, double *x, double *y, double *ans)
 {
  int t = (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x); // thread number of a thread inside a particular block
  int b = (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x); // block number of a block inside the grid
  int T = blockDim.x*blockDim.y*blockDim.z; // total number of blocks
  int B = gridDim.x*gridDim.y*gridDim.z; // total number of threads per block
  for(int i = b; i < N; i+=B) {
    for(int j = t; j < N; j+=T) {
      for(int k = 0; k < N; k++) {
        ans[i*N+j] += (x[i*N+k] * y[k*N+j]);
      }
    }
  }
 }
 
 // function to check whether we got correct value or not
 bool check(int N, double *ans)
 {
   for(int i = 0; i < N; i++) {
     for(int j = 0; j < N; j++) {
       if(ans[i*N+j] != (double)20.000000000000) return false;
     }
   }
   return true;
 }
 
 int main(void)
 {
   // size of matrix
   int N = 1<<9; // binary left-shift: 1 * 2^9 = 512
   printf("Size of matrix (N) is %d by %d.\n", N, N);
   int iter = 3;
   clock_t t;
   
   // Martices
   double *x, *y, *ans;
 
   // Allocate Unified Memory - accessible from both CPU and GPU
   cudaMallocManaged(&x, N*N*sizeof(double));
   cudaMallocManaged(&y, N*N*sizeof(double));
   cudaMallocManaged(&ans, N*N*sizeof(double));
   
    // Prefetch the data to the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N*N*sizeof(double), device, NULL);
    cudaMemPrefetchAsync(y, N*N*sizeof(double), device, NULL);
    cudaMemPrefetchAsync(ans, N*N*sizeof(double), device, NULL);

    int THREADS = 8;
    int BLOCKS = N / THREADS;
    dim3 threads(THREADS, THREADS); // we used dim3 to specify dimensions of vector type based on uint3
    dim3 blocks(BLOCKS, BLOCKS); // we used dim3 to specify dimensions of vector type based on uint3

    // initialize x,y and ans arrays on the host
    GPUmatmul_init<<<blocks, threads>>>(N, x, y, ans);
    
   // ..........................................................................
   double avg=0;
   std::cout<<"Starting Optimized GPU computation"<<std::endl;
   // Run kernel on GPU
   for(int i = 0; i <= iter; i++) {
     t = clock();
     // Launch kernel to multiply
     GPUmatmul<<<dim3(16,4,4),dim3(8,8,8)>>>(N, x, y, ans); // we used dim3 to specify dimensions of vector type based on uint3
     cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
     t = clock() - t;
     if(i) avg += t; //we will ignore the first run
     // printf ("It took GPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
   }
 
   avg /= iter;
   avg /= CLOCKS_PER_SEC;
   avg *= 1000;
   printf("It took %lf ms on avg.\n", avg);
   if(check(N,ans)) std::cout<<"RUN OK."<<std::endl; // Check for errors
   else std::cout<<"RUN NOT OK."<<std::endl;
 
   // Free memory
   cudaFree(x);
   cudaFree(y);
   cudaFree(ans);
 
   return 0;
 }
 /* EOF */