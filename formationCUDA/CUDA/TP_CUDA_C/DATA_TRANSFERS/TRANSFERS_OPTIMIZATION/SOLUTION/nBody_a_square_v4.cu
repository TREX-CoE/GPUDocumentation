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

float *x_gpu, *y_gpu, *z_gpu;
float *vx_gpu, *vy_gpu, *vz_gpu;
float *mass_gpu;



double wallclock(){
  struct timeval timer;
  gettimeofday(&timer, NULL);
  double time = timer.tv_sec + timer.tv_usec * 1.0E-6;
  return time;
}

#define BLOCKSIZE 256

//CUDA KERNELS
__global__ void update_velocity(int size,
				float* x , float* y , float* z ,
				float* vx, float* vy, float* vz,
				float *mass,
				float softeningSquared, float G,
				float timestep, float damping){


  float tmpax=0.0f;
  float tmpay=0.0f;
  float tmpaz=0.0f;
  
  int tx = threadIdx.x;
  int bx = blockIdx.x * blockDim.x;
  int index = tx + bx;

  int j;
  
  for (j = 0; j < size; j++)
    {
      float distance[3];
      float distanceSqr = 0.0f, distanceInv = 0.0f;
      distance[0] = x[j] - x[index];
      distance[1] = y[j] - y[index];
      distance[2] = z[j] - z[index];
      
      distanceSqr = sqrtf(distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2]) + softeningSquared;
      distanceInv = 1.0f / distanceSqr;
      
      float val = G * mass[j] * distanceInv * distanceInv * distanceInv;
      tmpax += distance[0] * val;
      tmpay += distance[1] * val;
      tmpaz += distance[2] * val;
    }
  // optimization use of scalars inside the j loop
  
  if( index < size){
    vx[index] = vx[index] + tmpax * timestep * damping;
    vy[index] = vy[index] + tmpay * timestep * damping;
    vz[index] = vz[index] + tmpaz * timestep * damping;
  }

}



__global__ void update_position(int size,
                                float* x , float* y , float* z ,
                                float* vx, float* vy, float* vz, 
				float timestep){

  int tx = threadIdx.x;
  int bx = blockIdx.x * blockDim.x;
  int index = tx + bx;

  if( index < size){
    x[index] += vx[index] * timestep;
    y[index] += vy[index] * timestep;
    z[index] += vz[index] * timestep;
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




void alloc_device_data()
{
  cudaMalloc((void **) &x_gpu , size * sizeof(float) );
  cudaMalloc((void **) &y_gpu , size * sizeof(float) );
  cudaMalloc((void **) &z_gpu , size * sizeof(float) );
  cudaMalloc((void **) &vx_gpu, size * sizeof(float) );
  cudaMalloc((void **) &vy_gpu, size * sizeof(float) );
  cudaMalloc((void **) &vz_gpu, size * sizeof(float) );
  cudaMalloc((void **) &mass_gpu, size * sizeof(float) );
}

void free_device_data()
{
  cudaFree(x_gpu);
  cudaFree(y_gpu);
  cudaFree(z_gpu);
  cudaFree(vx_gpu);
  cudaFree(vy_gpu);
  cudaFree(vz_gpu);
  cudaFree(mass_gpu);
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

  //alloc data on GPU
  alloc_device_data();


  double t0 = wallclock();
  cudaMemcpy(x_gpu , array_x , size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu , array_y , size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(z_gpu , array_z , size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vx_gpu, array_vx, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vy_gpu, array_vy, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vz_gpu, array_vz, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mass_gpu, array_mass, size * sizeof(float), cudaMemcpyHostToDevice);

  for (i=0; i<nb_steps; i++)
    {

      //-------------------//
      //  update velocity  //
      //-------------------//

      //update_velocity();     

      dim3 blocksize;
      dim3 gridsize;
      blocksize.x = BLOCKSIZE;
      gridsize.x  = (size + blocksize.x -1) / blocksize.x;

      update_velocity<<<gridsize,blocksize>>>(size, x_gpu, y_gpu, z_gpu , vx_gpu, vy_gpu, vz_gpu, mass_gpu, softeningSquared, G, timestep, damping);

      //-------------------//
      //  update position  //
      //-------------------//

      //update_position();  

      update_position<<<gridsize,blocksize>>>(size, x_gpu, y_gpu, z_gpu , vx_gpu, vy_gpu, vz_gpu, timestep);

    }

  cudaMemcpy(array_x , x_gpu , size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(array_y , y_gpu , size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(array_z , z_gpu , size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(array_vx, vx_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(array_vy, vy_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(array_vz, vz_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
  double t1 = wallclock();


#ifdef NORES
  printf("%d %lf %lf\n",size,timestep*nb_steps, t1-t0);
#else
  printf("size: %d     simulation time(in sec): %lf     execution time: %lf\n",size,timestep*nb_steps, t1-t0);
  check_res();
#endif

  //free GPU data
  free_device_data();
   
  free_host_data();
   
  return 0;
}
