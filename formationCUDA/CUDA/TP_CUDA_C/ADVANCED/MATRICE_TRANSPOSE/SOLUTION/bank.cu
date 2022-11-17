#include <stdio.h>
#include <stdlib.h>

//SIZE OF A CUDA WARP
#define WARP_SIZE 32
//NB_BANKS 16 for 1.x, 32 for >2.x
#define NB_BANKS 32


//must stay 32x32
#define BLOCK_X 32
#define BLOCK_Y 32


__global__ void myKernel(int size_x, int size_y, int* out){

  __shared__ int shmem[BLOCK_Y][BLOCK_X];


  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx =  tx + blockIdx.x * blockDim.x;
  int idy =  ty + blockIdx.y * blockDim.y;
  

  //write in shmen[ty][tx] the associated bank number. (shmem[0][0] = 0; ...; shmem[0][3] = 3 ...)
  //shmemt[ty][tx] = ... ;
  shmem[ty][tx] = (tx + ty * BLOCK_X) % NB_BANKS;

  __syncthreads();

  if( (idx<size_x) && (idy<size_y) ){
    out[idx + idy *size_x] = shmem[tx][ty];
  }
  
}

int main(int argc, char **argv){

  int i,j;

  int size_x = 512;
  int size_y = 512;

  int size_all = size_x * size_y;

  int *out = (int *) malloc(size_all * sizeof(int));

  for(i=0; i<size_all; i++){
    out[i] = 0;
  }


  int* d_out;

  cudaMalloc((void **)&d_out,size_all*sizeof(int));
  cudaMemcpy(d_out, out, size_all*sizeof(int), cudaMemcpyHostToDevice);
	
  dim3 dimBlock;
  dim3 dimGrid;

  dimBlock.x = BLOCK_X;
  dimBlock.y = BLOCK_Y;

  dimGrid.x = (size_x + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (size_y + dimBlock.y - 1) / dimBlock.y;

  myKernel<<<dimGrid, dimBlock>>>(size_x, size_y, d_out);
 
  cudaMemcpy(out, d_out, size_all*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_out);


  printf("bank accesses for the warps of the first block\n");
  for(j=0; j<BLOCK_Y; j++){
      printf("Warp %d:\n  ",j);
      for(i=0; i<BLOCK_X; i++){
	printf("%d ",out[i+j*size_x]);
      }
      printf("\n");
  }
  printf("\n");
  

  free(out);

  return 0;

}
