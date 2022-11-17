#include <stdio.h>
#include <stdlib.h>


__global__ void  copy2d(int size_x, int size_y, int *d_in, int *d_out){

  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  
  d_out[tx + ty *size_x] = d_in[tx + ty *size_x];

}


int main(int argc, char **argv){

  int size_x = 512;
  int size_y = 512;
  int size_all = size_x * size_y;

  //host allocation
  int* h_in  = (int *) malloc(size_all * sizeof(int));
  int* h_out = (int *) malloc(size_all * sizeof(int));


  //host initialization
  for(int i=0; i<size_all; i++){
    h_in [i] =  i;
    h_out[i] = -1;
  }


  //GPU allocation
  int *d_in;
  int *d_out;
  cudaMalloc( (void**) &d_in , size_all * sizeof(int) );
  cudaMalloc( (void**) &d_out, size_all * sizeof(int) );


  //copy h_in in d_in and h_out in d_out
  cudaMemcpy(d_in , h_in , size_all*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out, size_all*sizeof(int), cudaMemcpyHostToDevice);
  
  
  //kernel
  dim3 blocksize;
  dim3 gridsize;

  blocksize.x = 32;
  blocksize.y = 32;
  gridsize.x = (size_x + blocksize.x - 1) / blocksize.x;
  gridsize.y = (size_y + blocksize.y - 1) / blocksize.y;

  copy2d<<<gridsize, blocksize>>>(size_x, size_y, d_in, d_out);

  //copy d_out in h_out
  cudaMemcpy(h_out, d_out, size_all*sizeof(int), cudaMemcpyDeviceToHost);

  //check resulst
  int success = 1;
  int first_index_error = -1;

  for(int i=0; i<size_all; i++){
    //printf("%d %d\n", h_out[i], h_in[i]);
    if(h_out[i] != h_in[i] ){
      success = 0;
      first_index_error = i;
      break;
    }
  }

  if(success){
    printf("SUCCESS\n");
  }else{
    printf("ERROR: h_out[%d]=%d expected %d\n",first_index_error,h_out[first_index_error], h_in[first_index_error]);
  }


  //free GPU
  cudaFree(d_in );
  cudaFree(d_out);

  //free host
  free(h_in );
  free(h_out);

  return 0;

}
