#include <stdio.h>
#include <cuda.h>


//define mykernel as a GPU kernel
... void mykernel(){

  printf("Hello World\n");
}

int main(int argc, char **argv){

  //define a grid of 1 blocks and a block size of 4 threads (all on the x dimension)
  ... blocksize ...
  ... gridsize  ...

 
  printf("\nGrid: (%d,%d,%d) blocks.  Block: (%d,%d,%d) threads\n", 
	 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z);

  //launch kernel execution
  ...
  cudaDeviceSynchronize();





  //define a grid of 2 blocks and a block size of 4 threads (all on the x dimension)
  ... blocksize2 ...
  ... gridsize2  ...


  printf("\nGrid: (%d,%d,%d) blocks.  Block: (%d,%d,%d) threads\n", 
	 gridsize2.x, gridsize2.y, gridsize2.z, blocksize2.x, blocksize2.y, blocksize2.z);

  //launch kernel execution
  ...
  cudaDeviceSynchronize();


  return 0;
}
