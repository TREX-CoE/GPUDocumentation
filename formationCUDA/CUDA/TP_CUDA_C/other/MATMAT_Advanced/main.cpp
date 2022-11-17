#include <iostream>
#include <chrono> //high_resolution_clock
#include <iomanip>	//setprecision
#include <random>
#include <functional>
#include <cuda_profiler_api.h>

#include "test.h"

// ### forward declaration ###
/**
 * returns a random float
 */
static REAL getRandom();

/**
 * tranposes a sqared matrix in place
 */
static void transposeSqMat(REAL *);

/**
 * prints a NxN matrix
 */
static void printMat(const REAL *);

/**
 * resets all elements to zero
 */
static void clearMat(REAL *);

/**
 * compares elementwise, if matricies are equal within a the tolerance TOL
 */
static bool compareMat(const REAL *, const REAL *);

/**
 * computes the L2 matrix norm
 */
static float L2Norm(const REAL *, const REAL *);

/**
 * gives an estimate (upper bound) for the condition number of the matrix M
 */
static float getConditionNumber(const REAL *);

/**
 * prints CEPP splash screen
 */
static void printSplashScreen();

//main program
int main(int argc, char ** argv){
	printSplashScreen();

	//setup GPU (this has to be done, before memory is allocated on the device)
	cudaSetDeviceFlags(cudaDeviceMapHost);
	
	//Tests:
	std::cout << ((testA)?"Test A is active.":"Test A is inactive.") << "\n";
	std::cout << ((testB)?"Test B is active.":"Test B is inactive.") << "\n";
	std::cout << ((testC)?"Test C is active.":"Test C is inactive.") << "\n";
	std::cout << ((testD)?"Test D is active.":"Test C is inactive.") << "\n";
	
	REAL *A, *B, *C1, *C2, *C3, *C4;
	long long timings[4];	//in ys
	char check[3];			//
	float norms[3];			//matrix L2 norms
	REAL garbage[4]; 		//keeps track of the lib init (so that they do not get optimized away)
	for(int i = 0; i < 3; ++i)
		timings[i] = 0ULL;
	
	A = new REAL[N * N];
	B = new REAL[N * N];
	C1 = new REAL[N * N];
	C2 = new REAL[N * N];
	//C3 = new REAL[N * N];
	//C4 = new REAL[N * N];
	cudaHostAlloc((void **)&C3, N * N * sizeof(REAL), cudaHostAllocMapped);
	cudaHostAlloc((void **)&C4, N * N * sizeof(REAL), cudaHostAllocMapped);
	
	
	for(int i = 0; i < N * N; ++i){
		A[i] = (getRandom() + 1) * 1; // +1 to ensure matrix is well-conditioned
		B[i] = (getRandom() + 1) * 1;
	}
	clearMat(C1);
	clearMat(C2);
	clearMat(C3);
	clearMat(C4);
	
	if(testA){
		Test *tst = new TestA();
		tst->initLibs(&(garbage[0]));
		
		auto start = std::chrono::high_resolution_clock::now();
		
		tst->execTest(A,B,C1);
		
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		timings[0] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		delete tst; 
	}
	
	if(testB){
		Test *tst = new TestB();
		tst->initLibs(&(garbage[1]));
		auto start = std::chrono::high_resolution_clock::now();
		
		tst->execTest(A,B,C2);
		
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		timings[1] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		delete tst; 
	}
	
	
	
	if(testC){
		Test *tst = new TestC();
		tst->initLibs(&(garbage[2]));

		auto start = std::chrono::high_resolution_clock::now();
		
		tst->execTest(A,B,C3);
		
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		timings[2] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		
		//transposeSqMat(C3);
		
		delete tst; 
	}
	
	cudaProfilerStart();
	
	if(testD){
		Test *tst = new TestD();
		tst->initLibs(&(garbage[3]));
		auto start = std::chrono::high_resolution_clock::now();
		std::cout << "\n\n";
		tst->execTest(A,B,C4);
		//dynamic_cast<TestD *>(tst)->stagedTransposeMat(A, C1);
		
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		timings[3] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		delete tst; 
	}
	
	cudaProfilerStop();
	
	if(N<1025){
		if(testA && testB) check[0] = (compareMat(C1,C2)) ? 't' : 'f';
		if(testA && testC) check[1] = (compareMat(C1,C3)) ? 't' : 'f';
		if(testA && testD) check[2] = (compareMat(C1,C4)) ? 't' : 'f';
		
		if(testA && testB) norms[0] = L2Norm(C1,C2);
		if(testA && testC) norms[1] = L2Norm(C1,C3);
		if(testA && testD) norms[2] = L2Norm(C1,C4);
	}else{
		std::cout << "N too large, no norms or correctness checks are done (N > 1024)\n";
	}
	
	//print stats
	if(testA)
		std::cout << "Test A took " << timings[0] << " microseconds \n";
	if(testB)
		std::cout << "Test B took " << timings[1] << " microseconds, L2 difference: " << norms[0] << " (" << check[0] << ")\n";
	if(testC)
		std::cout << "Test C took " << timings[2] << " microseconds, L2 difference: " << norms[1] << " (" << check[1] << ")\n";
	if(testD)
		std::cout << "Test D took " << timings[3] << " microseconds, L2 difference: " << norms[2] << " (" << check[2] << ")\n";
	
	
	delete [] A;
	delete [] B;
	delete [] C1;
	delete [] C2;
	//delete [] C3;
	//delete [] C4;
	cudaFree(C3);
	cudaFree(C4);


	//std::cout << "(init values : " << garbage[0] << " " << garbage[1] << " " << garbage[2] << " " << garbage[3] <<	" )\n";
	std::cout << "END OF PROG\n";
	
	return EXIT_SUCCESS;
}

//helper functions

REAL getRandom(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return (REAL) dis(e);
}

void transposeSqMat(REAL *M){
	for(int i = 0; i < N; ++i){
		for(int j = i; j < N; ++j){
			REAL tmp = M[j * N + i];
			M[j * N + i] = M[i * N + j];
			M[i * N + j] = tmp;
		}
	}
}


void printMat(const REAL *M){
	for(int i = 0; i < N; ++i){
		for(int j = 0; j < N; ++j){
			std::cout << std::setprecision(3) << M[i * N + j] << " ";
		}
		std::cout << std::endl;
	}
}

void clearMat(REAL * M){
	for(int i = 0; i < (N*N); ++i){
		M[i] = .0f;
	}
}

bool compareMat(const REAL * M1, const REAL * M2){
	for(int i = 0; i < (N*N); ++i){
		if(fabs(M1[i] - M2[i]) > TOL)
			return false;
	}
	return true;
}

float getConditionNumber(const REAL *M){
	float min = 1.e5f, max = -1e5f;
	for(int i = 0; i < (N*N); ++i){
		min = (M[i] < min) ? M[i] : min;
		max = (M[i] > max) ? M[i] : max;
	}
	return fabs(max) / fabs(min);
}

float L2Norm(const REAL *A, const REAL *B){
	float res = .0f;
	for(int i = 0; i < (N*N); ++i){
		res += fabs(A[i] - B[i]) * fabs(A[i] - B[i]);
	}
	std::cout << res << "\n";
	return sqrt(res);
}

void printSplashScreen(){	
	std::cout << 
	"##################################################\n" <<
	"######--------##--------##--------##--------######\n" <<
	"######--########--########--####--##--####--######\n" <<
	"######--########--########--####--##--####--######\n" <<
	"######--########--########--####--##--####--######\n" <<
	"######--########-----#####--------##--------######\n" <<
	"######--########--########--########--############\n" <<
	"######--########--########--########--############\n" <<
	"######--########--########--########--############\n" <<
	"######--########--########--########--############\n" <<
	"######--------##--------##--########--############\n" <<
	"##################################################\n" <<
	"# CENTER OF EXCELLENCE FOR PERFORMANCE COMPUTING #\n" <<
	"##################################################\n\n";
}
