#include <stdio.h>
#include <cuda.h>

__global__ void mykernel(){

  printf("Hello World\n");
}

int main(int argc, char **argv){

  //define a grid of 1 blocks and a block size of 4 threads (all on the x dimension)
  dim3 blocksize(4,1,1);
  dim3 gridsize(1,1,1);
  
  printf("\nGrid: (%d,%d,%d) blocks.  Block: (%d,%d,%d) threads\n", 
	 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z);
  //launch kernel execution
  mykernel<<<gridsize, blocksize>>>();
  cudaDeviceSynchronize();





  //define a grid of 2 blocks and a block size of 4 threads (all on the x dimension)
  dim3 blocksize2(4,1,1);
  dim3 gridsize2(2,1,1);

  printf("\nGrid: (%d,%d,%d) blocks.  Block: (%d,%d,%d) threads\n", 
	 gridsize2.x, gridsize2.y, gridsize2.z, blocksize2.x, blocksize2.y, blocksize2.z);

  //launch kernel execution
  mykernel<<<gridsize2, blocksize2>>>();
  cudaDeviceSynchronize();


  return 0;
}
