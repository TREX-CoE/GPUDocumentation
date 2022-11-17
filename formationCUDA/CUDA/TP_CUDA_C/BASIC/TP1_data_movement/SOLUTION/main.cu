#include <stdio.h>
#include <stdlib.h>


#define CudaSafeCall( err )   __cudaSafeCall( err, __FILE__, __LINE__ )

void __cudaSafeCall( cudaError err, const char *file, const int line ){
#if defined(DEBUG) || defined(_DEBUG)
     if ( cudaSuccess != err ) 
    {
     fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",file, line, cudaGetErrorString(err));
     exit( err );
     }
#endif
}



int main(int argc, char **argv){

  int i;
  int size = 256;

  //host allocation
  int* h_a = (int *) malloc(size * sizeof(int));
  int* h_b = (int *) malloc(size * sizeof(int));

  //host initialization
  for(i=0; i<size; i++){
    h_a[i] = i;
    h_b[i] = 0;
  }


  //GPU allocation
  int *d_a;
  int *d_b;
  cudaMalloc( (void**) &d_a, size * sizeof(int) );
  cudaMalloc( (void**) &d_b, size * sizeof(int) );

  //copy h_a in d_a
  cudaMemcpy(d_a, h_a, size*sizeof(int), cudaMemcpyHostToDevice);
  
  //copy d_a in d_b
  cudaMemcpy(d_b, d_a, size*sizeof(int), cudaMemcpyDeviceToDevice);

  //copy d_b in h_b
  cudaMemcpy(h_b, d_b, size*sizeof(int), cudaMemcpyDeviceToHost);

  //check resulst
  int success = 1;
  for(i=0; i<size; i++){
    if(h_b[i] != h_a[i])
      success = 0;
  }

  if(success){
    printf("SUCCESS\n");
  }else{
    printf("ERROR\n");
  }


  //free GPU
  cudaFree(d_a);
  cudaFree(d_b);

  //free host
  free(h_a);
  free(h_b);

  return 0;
}
