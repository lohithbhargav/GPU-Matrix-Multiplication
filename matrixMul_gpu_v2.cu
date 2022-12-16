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
 
 // Kernel function to multiple the elements of two arrays
 __global__
 void GPUmatmul(int N, double *x, double *y, double *ans)
 {
  // Calculate the index for each thread row and column using blockDim
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for(int k = 0; k < N; k++) {
    ans[index] += (x[blockIdx.x*blockDim.x+k] * y[k*blockDim.y+threadIdx.y]);
  }
 }
 
 // function to check whether we got correct value or not
 bool check(int N, double *ans)
 {
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(ans[i*N+j] != 20.0) return false;
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
   
   // initialize x,y and ans arrays on the host
   for (int i = 0; i < N; i++) {
     for(int j = 0; j < N; j++) {
       x[i*N+j] = 5;
       y[i*N+j] = (i==j?1:0);
       ans[i*N+j] = (double)0.000000000000;
     }
   }
 
   double avg=0;
   std::cout<<"Starting Optimized GPU computation"<<std::endl;
   // Run kernel on GPU
   for(int i = 0; i <= iter; i++) {
     t = clock();
     GPUmatmul<<<512,512>>>(N, x, y, ans); // we have updated the threads and blocks from v1
     cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
     t = clock() - t;
     if(i) avg += t; //we will ignore the first run
     // printf ("It took GPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
   }
   
   avg /= iter;
   avg /= CLOCKS_PER_SEC;
   avg *= 1000;
   printf("It took %lf ms on avg.\n", avg);
   if(check(N,ans)){
    std::cout<<"RUN OK."<<std::endl; // Check for errors
    }
   else{
    std::cout<<"RUN NOT OK."<<std::endl;
    }
 
   // Free memory
   cudaFree(x);
   cudaFree(y);
   cudaFree(ans);
 
   return 0;
 }
 /* EOF */