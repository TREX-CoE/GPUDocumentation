#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

//must stay 32x32
#define BLOCK_X 32
#define BLOCK_Y 32

extern "C" __global__ void myKernel_ref(int size_x, int size_y, int size_z, int* in, int* out){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = tx + blockIdx.x * blockDim.x;
  int idy = ty + blockIdx.y * blockDim.y;
  int idz = 0;

  int temp;

  for(idz = 4; idz < (size_z-4); idz++){

    temp = in[idx + idy *size_x + (idz+0) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+1) * size_x * size_y] - in[idx + idy *size_x + (idz-1) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+2) * size_x * size_y] - in[idx + idy *size_x + (idz-2) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+3) * size_x * size_y] - in[idx + idy *size_x + (idz-3) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+4) * size_x * size_y] - in[idx + idy *size_x + (idz-4) * size_x * size_y];

    if( (idx<size_x) && (idy<size_y) ){
      out[idx + idy *size_x + idz * size_x * size_y] = temp;
    }

  }

}


extern "C" __global__ void myKernel_restrict(int size_x, int size_y, int size_z, const int* __restrict__ in, int* __restrict__ out){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = tx + blockIdx.x * blockDim.x;
  int idy = ty + blockIdx.y * blockDim.y;
  int idz = 0;

  int temp;
  
  for(idz = 4; idz < (size_z-4); idz++){

    temp = in[idx + idy *size_x + (idz+0) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+1) * size_x * size_y] - in[idx + idy *size_x + (idz-1) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+2) * size_x * size_y] - in[idx + idy *size_x + (idz-2) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+3) * size_x * size_y] - in[idx + idy *size_x + (idz-3) * size_x * size_y]
      +    in[idx + idy *size_x + (idz+4) * size_x * size_y] - in[idx + idy *size_x + (idz-4) * size_x * size_y];

    if( (idx<size_x) && (idy<size_y) ){
      out[idx + idy *size_x + idz * size_x * size_y] = temp; 
    }

  }
  
}


extern "C" __global__ void myKernel_pipeline(int size_x, int size_y, int size_z, const int* __restrict in, int* __restrict__ out){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = tx + blockIdx.x * blockDim.x;
  int idy = ty + blockIdx.y * blockDim.y;
  int idz = 0;


  //load in register the first memory accesses
  //initalization of the pipeline
  int in_m4 = ...
  int in_m3 = ...
  int in_m2 = ...
  int in_m1 = ...
  int in_cu = ...
  int in_p1 = ...
  int in_p2 = ...
  int in_p3 = ...
  int in_p4 = ...
  

  for(idz = 4; idz < (size_z-4); idz++){

    //pipeline
    in_m4 = ...
    in_m3 = ...
    in_m2 = ...
    in_m1 = ...
    in_cu = ...
    in_p1 = ...
    in_p2 = ...
    in_p3 = ...
    in_p4 = ...
    
    int temp = in_cu
          +    ...
          +    ...
          +    ...
          +    ...

    if( (idx<size_x) && (idy<size_y) ){
      out[idx + idy *size_x + idz * size_x * size_y] = temp; 
    }

  }
  
}




int main(int argc, char **argv){

  int i;

  int size_x = 512;
  int size_y = 512;
  int size_z = 512;

  int size_all = size_x * size_y * size_z;

  int *in  = (int *) malloc(size_all * sizeof(int));
  int *out1 = (int *) malloc(size_all * sizeof(int));
  int *out2 = (int *) malloc(size_all * sizeof(int));
  int *out3 = (int *) malloc(size_all * sizeof(int));

  for(i=0; i<size_all; i++){
    in  [i] = i;
    out1[i] = 0;
    out2[i] = 0;
    out3[i] = 0;
  }


  //CUDA PART
  int* d_in;
  int* d_out1;
  int* d_out2;
  int* d_out3;


  cudaMalloc((void **)&d_in  ,size_all*sizeof(int));
  cudaMalloc((void **)&d_out1,size_all*sizeof(int));
  cudaMalloc((void **)&d_out2,size_all*sizeof(int));
  cudaMalloc((void **)&d_out3,size_all*sizeof(int));  

  cudaMemcpy(d_in  , in  , size_all*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out1, out1, size_all*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out2, out2, size_all*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out3, out3, size_all*sizeof(int), cudaMemcpyHostToDevice);


  dim3 dimBlock;
  dim3 dimGrid;

  dimBlock.x = BLOCK_X;
  dimBlock.y = BLOCK_Y;

  dimGrid.x = (size_x + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (size_y + dimBlock.y - 1) / dimBlock.y;

  //WARMUP
  myKernel_ref     <<<dimGrid, dimBlock>>>(size_x, size_y, size_z, d_in, d_out1);
  myKernel_restrict<<<dimGrid, dimBlock>>>(size_x, size_y, size_z, d_in, d_out2);
  myKernel_pipeline<<<dimGrid, dimBlock>>>(size_x, size_y, size_z, d_in, d_out3);

  //RUN
  myKernel_ref     <<<dimGrid, dimBlock>>>(size_x, size_y, size_z, d_in, d_out1);
  myKernel_restrict<<<dimGrid, dimBlock>>>(size_x, size_y, size_z, d_in, d_out2);
  myKernel_pipeline<<<dimGrid, dimBlock>>>(size_x, size_y, size_z, d_in, d_out3);


  cudaMemcpy(out1, d_out1, size_all*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(out2, d_out2, size_all*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(out3, d_out3, size_all*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_in );
  cudaFree(d_out1);
  cudaFree(d_out2);
  cudaFree(d_out3);

  //CHECK RESULTS
  for(i=0; i<size_all; i++){

    if(out2[i] != out1[i]){
      printf("error out1[%d] != out2[%d] : %d %d\n", i,i,out1[i],out2[i]);
      return -1;
    }
    if(out3[i] != out1[i]){
      printf("error out1[%d] != out3[%d] : %d %d\n", i,i,out1[i],out3[i]);
      return -1;
    }

  }


  free(in );
  free(out1);
  free(out2);
  free(out3);

  return 0;

}
