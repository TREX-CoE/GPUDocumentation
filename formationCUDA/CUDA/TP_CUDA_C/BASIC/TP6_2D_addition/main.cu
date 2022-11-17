#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"


__global__ void  add2d(int size_x, int size_y, int *d_a, int *d_b, int *d_c){

	//compute d_c = d_a + d_b
	int index,i_x,i_y;
	
	i_x = threadIdx.x + blockIdx.x * blockDim.x;
	i_y = threadIdx.y + blockIdx.y * blockDim.y;

	index = i_x + i_y * size_x;

	if(i_x < size_x && index_y < size_y)
		d_c[index] = d_a[index] + d_b[index];
}


int main(int argc, char **argv){

	int i;
	int size_x = 64;
	int size_y = 8;
	int size_all = size_x * size_y;

	//host allocation
	int* h_a = (int *) malloc(size_all * sizeof(int));
	int* h_b = (int *) malloc(size_all * sizeof(int));
	int* h_c = (int *) malloc(size_all * sizeof(int));

	//host initialization
	for(i=0; i<size_all; i++){
		h_a[i] = i;
		h_b[i] = 2*i;
		h_c[i] = 0;
	}


	//GPU allocation
	int *d_a;
	int *d_b;
	int *d_c;
	cudaMalloc( (void**) &d_a, size_all * sizeof(int) );
	cudaMalloc( (void**) &d_b, size_all * sizeof(int) );
	cudaMalloc( (void**) &d_c, size_all * sizeof(int) );

	//copy h_a in d_a and h_b in d_b
	cudaMemcpy(d_a, h_a, size_all*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size_all*sizeof(int), cudaMemcpyHostToDevice);


	//kernel
	dim3 blocksize;
	dim3 gridsize;

	blocksize.x = 32;
	blocksize.y =  4;
	gridsize.x = (size_x + blocksize.x-1) / blocksize.x;
	gridsize.x = (size_y + blocksize.y-1) / blocksize.y;

	add2d<<<gridsize, blocksize>>>(size_x, size_y, d_a, d_b, d_c);

	//copy d_c in h_c
	cudaMemcpy(h_c, d_c, size_all*sizeof(int), cudaMemcpyDeviceToHost);

	//check resulst
	int success = 1;
	int first_index_error = -1;

	for(i=0; i<size_all; i++){
		//printf("%d %d\n", h_c[i], 3*i);
		if(h_c[i] != (3*i) )
			success = 0;
		first_index_error = i;
		break;
	}
}

if(success){
	printf("SUCCESS\n");
}else{
	printf("ERROR: h_c[%d]=%d expected %d\n",first_index_error,h_c[first_index_errors], 3*first_index_error);
}


//free GPU
cudaFree(d_a);
cudaFree(d_b);

//free host
free(h_a);
free(h_b);

}
