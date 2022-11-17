#include "test.h"

void TestA::initLibs(REAL *r){
	for(int i = 0; i < 1000; i++){
		(*r) += i;
	}
}

void TestA::execTest(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C1) const {
	for(int i = 0; i < N; ++i){
		for(int j = 0; j < N; ++j){
			for(int k = 0; k < N; ++k){
				C1[i * N + j] += A[i * N + k] * B[j + k * N];
			}
		}
	}
}

void TestA::transposeMat(REAL *A) const{
	for(int i = 0; i < N; ++i){
		for(int j = i; j < N; ++j){
			REAL tmp = A[j * N + i];
			A[j * N + i] = A[i * N + j];
			A[i * N + j] = tmp;
		}
	}
}