#include "test.h"

//#include <cblas.h> //BLAS
#include <iostream>
#include <math.h>

void TestB::initLibs(REAL *r){
	REAL x[16] = {0, 0, 0, 0, 0, 0, 0, 0, -28, 0, 41, -15, 0, 20, -55.744289, -76.930283};
	REAL y[16] = {0, 0, 0, 0, 0, 0, 0, 0, -54.654705, 0, 23.670898, -4.533447, 0, 79.964569, -65.6026, -4};

	
	REAL expected = .0f;
    for (int i = 0; i < 16; i++) {
        expected += x[i] * y[i];
    }

    REAL actual = /*cblas_sdot(16, x, 1, y, 1)*/ .0;
    (*r) = fabs(expected -  actual);
}

void TestB::execTest(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C) const {
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}
