#include <iostream>
//cublas and cuda
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


//macros
#define REAL double //be aware that changing this to float requires to call the appropriate Blas routines for single precision!

//variables
const int N = 4096;
const float TOL = 1e-6f;	
const bool testA = false;
const bool testB = false;
const bool testC = true;
const bool testD = true;


class Test{	
public:
	Test() {};
	virtual ~Test() {};
	
	virtual void initLibs(REAL *) = 0;
	
	virtual void execTest(const REAL *, const REAL *, REAL *) const = 0;
};

class TestA : public Test{
private:
	void transposeMat(REAL *A) const;
public:
	virtual void initLibs(REAL *);

	virtual void execTest(const REAL *, const REAL *, REAL *) const;
};

class TestB : public Test{
public:
	virtual void initLibs(REAL *);

	virtual void execTest(const REAL *, const REAL *, REAL *) const;
};

class TestC : public Test{
private:
	cublasHandle_t *handle = nullptr;
	
	void printCudaStats();
public:
	virtual void initLibs(REAL *);

	virtual void execTest(const REAL *, const REAL *, REAL *) const;
};

class TestD : public Test{
private:
	void naiveImpl(const REAL *, const REAL *, REAL *) const;
	
	void stagedLoad(const REAL *, const REAL *, REAL *) const;
	
	void zeroCopy(const REAL *, const REAL *, REAL *) const;
	
public:
	virtual void initLibs(REAL *);
	
	void transposeMat(const REAL * A, REAL * C) const;
	
	virtual void execTest(const REAL *, const REAL *, REAL *) const;
	
	virtual void stagedTransposeMat(const REAL *, REAL *) const;
};
