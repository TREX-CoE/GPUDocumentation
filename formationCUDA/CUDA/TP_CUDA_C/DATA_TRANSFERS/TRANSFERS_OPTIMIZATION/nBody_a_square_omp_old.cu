#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>


#define M_PI 3.14159265358979323846 
const float softeningSquared = 0.001215000f*0.001215000f;
const float G = 6.67259e-11f;
const float timestep = 1.0f;
const float damping = 1.0f;


int size;

float *array_x, *array_y, *array_z; 
float *array_vx, *array_vy, *array_vz;
float *array_mass;

//Device variables

float *d_array_x, *d_array_y, *d_array_z; 
float *d_array_vx, *d_array_vy, *d_array_vz;
float *d_array_mass;



double wallclock(){
	struct timeval timer;
	gettimeofday(&timer, NULL);
	double time = timer.tv_sec + timer.tv_usec * 1.0E-6;
	return time;
}


__global__ void ud_velo_GPU(){

	for (int i = 0; i < size; i++)// update velocity
	{
		int j;

		// optimization use of scalars inside the j loop for acceleration
		float tmpax=0.0f;
		float tmpay=0.0f;
		float tmpaz=0.0f;

		for (j = 0; j < size; j++)
		{
			float distance[3];
			float distanceSqr = 0.0f, distanceInv = 0.0f;
			distance[0] = array_x[j] - array_x[i];
			distance[1] = array_y[j] - array_y[i];
			distance[2] = array_z[j] - array_z[i];

			distanceSqr = sqrtf(distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2]) + softeningSquared;
			distanceInv = 1.0f / distanceSqr;

			float val = G * array_mass[j] * distanceInv * distanceInv * distanceInv;
			tmpax += distance[0] * val;
			tmpay += distance[1] * val;
			tmpaz += distance[2] * val;
		}
		// optimization use of scalars inside the j loop

		float vx = array_vx[i] + tmpax * timestep * damping;
		float vy = array_vy[i] + tmpay * timestep * damping;
		float vz = array_vz[i] + tmpaz * timestep * damping;


		array_vx[i] = vx;
		array_vy[i] = vy;
		array_vz[i] = vz;
	}


}

void update_velocity()
{

#pragma omp parallel for
	for (int i = 0; i < size; i++)// update velocity
	{
		int j;

		// optimization use of scalars inside the j loop for acceleration
		float tmpax=0.0f;
		float tmpay=0.0f;
		float tmpaz=0.0f;

		for (j = 0; j < size; j++)
		{
			float distance[3];
			float distanceSqr = 0.0f, distanceInv = 0.0f;
			distance[0] = array_x[j] - array_x[i];
			distance[1] = array_y[j] - array_y[i];
			distance[2] = array_z[j] - array_z[i];

			distanceSqr = sqrtf(distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2]) + softeningSquared;
			distanceInv = 1.0f / distanceSqr;

			float val = G * array_mass[j] * distanceInv * distanceInv * distanceInv;
			tmpax += distance[0] * val;
			tmpay += distance[1] * val;
			tmpaz += distance[2] * val;
		}
		// optimization use of scalars inside the j loop

		float vx = array_vx[i] + tmpax * timestep * damping;
		float vy = array_vy[i] + tmpay * timestep * damping;
		float vz = array_vz[i] + tmpaz * timestep * damping;


		array_vx[i] = vx;
		array_vy[i] = vy;
		array_vz[i] = vz;
	}

}


void update_position()
{

#pragma omp parallel for
	for (int i = 0; i < size; i++) // update position
	{
		array_x[i] += array_vx[i] * timestep;
		array_y[i] += array_vy[i] * timestep;
		array_z[i] += array_vz[i] * timestep;

	}  
}
void alloc_device_data()
{
	cudaMalloc((void**)&d_array_x, size * sizeof(float) );
	cudaMalloc((void**)&d_array_y, size * sizeof(float) );
	cudaMalloc((void**)&d_array_vx, size * sizeof(float) );
	cudaMalloc((void**)&d_array_vy, size * sizeof(float) );
	cudaMalloc((void**)&d_array_vz, size * sizeof(float) );
	cudaMalloc((void**)&d_array_mass, size * sizeof(float) );

}

void alloc_host_data()
{
	array_x    = (float *) malloc( size * sizeof(float) );
	array_y    = (float *) malloc( size * sizeof(float) );
	array_z    = (float *) malloc( size * sizeof(float) );
	array_vx   = (float *) malloc( size * sizeof(float) );
	array_vy   = (float *) malloc( size * sizeof(float) );
	array_vz   = (float *) malloc( size * sizeof(float) );
	array_mass = (float *) malloc( size * sizeof(float) );
}
void free_device_data()
{
	cudaFree(array_x);
	cudaFree(array_y);
	cudaFree(array_z);
	cudaFree(array_vx);
	cudaFree(array_vy);
	cudaFree(array_vz);
	cudaFree(array_mass);
}
void free_host_data()
{
	free(array_x);
	free(array_y);
	free(array_z);
	free(array_vx);
	free(array_vy);
	free(array_vz);
	free(array_mass);
}

void init_host_data()
{
	int i = 0;


	//Earth
	const double earthmass = 5.972e24;
	array_x[0] = 0.0f;
	array_y[0] = 0.0f;
	array_z[0] = 0.0f;
	array_vx[0] = 0.0f;
	array_vy[0] = 0.0f;
	array_vz[0] = 0.0f;
	array_mass[0] = (float) earthmass;

	double angle = 2.0 * M_PI / (size-1) * 0.5; //half circle
	double radius = 7000 * 1e3; //(600 + 6400) * 1e3; //60km
	double vtan  = sqrt(G*earthmass/radius);
	double anglez = M_PI/8.0;

#ifndef NORES
	printf("vtan: %lf\n", vtan);
#endif

	for (i = 1; i < size; i++)
	{     
		double x = radius * cos(angle * (i-1)) * cos(anglez);
		double y = radius * sin(angle * (i-1));
		double z = radius * cos(angle * (i-1)) * sin(anglez);

		double vx = -vtan * sin(angle * (i-1)) * cos(anglez); //cos(PI/2 + theta) = -sin(theta)
		double vy =  vtan * cos(angle * (i-1));               //sin(PI/2 + theta) =  cos(theta)
		double vz = -vtan * sin(angle * (i-1)) * sin(anglez);

		array_x[i]  = (float) x;
		array_y[i]  = (float) y;
		array_z[i]  = (float) z;
		array_vx[i] = (float) vx;
		array_vy[i] = (float) vy;
		array_vz[i] = (float) vz;
		array_mass[i] = 1e4f;
	}

}


void check_res()
{
	int i;

	double result_x  = 0.0;
	double result_y  = 0.0;
	double result_z  = 0.0;
	double result_vx = 0.0;
	double result_vy = 0.0;
	double result_vz = 0.0;
	double result_mass = 0.0;

	for (i = 0; i < size; i++)
	{
		result_x  += array_x[i];
		result_y  += array_y[i];
		result_z  += array_z[i];
		result_vx += array_vx[i];
		result_vy += array_vy[i];
		result_vz += array_vz[i];
		result_mass += array_mass[i];
	}

	printf("RES: %g %g %g %g %g %g\n", result_x, result_y, result_z, result_vx, result_vy, result_vz);

	double maxvtan = 0.0;
	double minvtan = 1e100;

	for (i = 1; i < size; i++)
	{
		double vittan = sqrt(array_vx[i] *  array_vx[i] + array_vy[i] * array_vy[i]  + array_vz[i] * array_vz[i]);
		maxvtan = fmax(maxvtan,vittan);
		minvtan = fmin(minvtan,vittan);
	}

	printf("maxvtan: %lf    minvtan: %lf\n",maxvtan, minvtan);

}



int main(int argc, char* a[])
{

	int i = 0;
	int nb_steps = 200;
	size = 100;

	if(argc > 1){
		size = atoi(a[1]);
		if(size < 2){
			printf("size must be at least 2\n");
			exit(-1);
		}
	}  
	if (argc > 2){
		nb_steps = atoi(a[2]);
		if(nb_steps < 0){
			printf("nb_steps must be greater than 0\n");
			exit(-1);
		}
	}


	alloc_host_data();

	init_host_data();


	double t0 = wallclock();
	for (i=0; i<nb_steps; i++)
	{
		update_velocity();
		update_position();
	}
	double t1 = wallclock();

	//some more checks
	//for(int i = 0; i < size; i=i+10){
	//	printf("array_x: %f\n", array_x[i]);
	//}

#ifdef NORES
	printf("%d %lf %lf\n",size,timestep*nb_steps, t1-t0);
#else
	printf("size: %d     simulation time(in sec): %lf     execution time: %lf\n",size,timestep*nb_steps, t1-t0);
	check_res();
#endif

	free_host_data();

	return 0;
}
