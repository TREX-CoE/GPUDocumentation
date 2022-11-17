#include <stdio.h>


__device__ int x;


__global__ void ok_kernel(int *a) {
	a[0] = 10;
}
__global__ void bad_kernel(int *a) {
	a[42] = 10;
	int b = a[0];
}
int main() {
	int *array = (int *) malloc(sizeof(int) * 10);
	int *d_array;
	
	cudaMalloc((void **) &d_array, sizeof(int) * 10);
	
	
	cudaMemcpy(d_array, array, sizeof(int) * 10, cudaMemcpyHostToDevice);
	
	ok_kernel<<<1,1>>>(d_array);
	bad_kernel<<<1,1>>>(d_array);
	
	cudaMemcpy(array, d_array, sizeof(int) * 10, cudaMemcpyDeviceToHost);
	printf("%d \n", array[0]);
	
	return 0;
}
