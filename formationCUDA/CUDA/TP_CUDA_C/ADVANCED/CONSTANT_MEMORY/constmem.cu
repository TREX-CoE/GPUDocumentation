#include <stdio.h>
#include <stdlib.h>


__constant__ float coeff[4];


__global__ void mykernel(float *a, float *coef){
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;

  for(int i=0; i<4; i++)
    a[index] = a[index] + coef[i];

}




__global__ void mykernel_constmem(float *a){
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;

  for(int i=0; i<4; i++)
    a[index] = a[index] + coeff[i];

}


int main(int argc, char ** argv){

  float cpucoeff[4] = {11.32f, 65.36f,  8.32f, 47.28f};

  //initialization on the host
  int size = 1024* 1024 * 128;
  float *h_a    = (float *) malloc(size * sizeof(float));
  float *h_res1 = (float *) malloc(size * sizeof(float));
  float *h_res2 = (float *) malloc(size * sizeof(float));

  for(int i=0; i<size; i++){
    h_a[i] = (float) i;
  }


  //kernel without constmem
  float *d_a1;
  cudaMalloc( (void **) &d_a1, size *sizeof(float));
  cudaMemcpy(d_a1, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
  
  
  float *d_coef;
  cudaMalloc( (void **) &d_coef, 4 *sizeof(float));
  cudaMemcpy(d_coef, &cpucoeff, 4 *sizeof(float), cudaMemcpyHostToDevice);


  dim3 blocksize(128);
  dim3 gridsize;
  gridsize.x = (size + blocksize.x -1) / blocksize.x;
  
  mykernel<<<gridsize,blocksize>>>(d_a1, d_coef);
  mykernel<<<gridsize,blocksize>>>(d_a1, d_coef);
  
  cudaMemcpy(h_res1, d_a1, size*sizeof(float), cudaMemcpyDeviceToHost);
  printf("mykernel: expected value: %lf,  value: %lf\n",  22.f + 2*(11.32f+ 65.36f+ 8.32f+ 47.28f), h_res1[22]);  
  cudaFree(d_a1);
  
  
  
  //kernel with constem
  float *d_a2;
  cudaMalloc( (void **) &d_a2, size *sizeof(float));
  cudaMemcpy(d_a2, h_a, size*sizeof(float), cudaMemcpyHostToDevice);

  //copy cpucoeff in const memory
  cudaMemcpyFromSymbol(coeff, cpucoeff, 4*sizeof(float),0,cudaMemcpyHostToDevice);
 
  
  mykernel_constmem<<<gridsize,blocksize>>>(d_a2);
  mykernel_constmem<<<gridsize,blocksize>>>(d_a2);
    
  cudaMemcpy(h_res2, d_a2, size*sizeof(float), cudaMemcpyDeviceToHost);
  printf("mykernel_constmem: expected value: %lf,  value: %lf\n",  22.f + 2*(11.32f+ 65.36f+ 8.32f+ 47.28f), h_res2[22]);  
  cudaFree(d_a2);
  
  free(h_a);
  free(h_res1);
  free(h_res2);
  
  return 0;

