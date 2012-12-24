#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "complex_h.cuh"

__host__ __device__ complex complex_from_polar(double r, double theta_radians) {
	complex result;
	result.re = r * cos(theta_radians);
	result.im = r * sin(theta_radians);
	return result;
}
__host__ __device__ complex complex_add(complex left, complex right) {
	complex result;
	result.re = left.re + right.re;
	result.im = left.im + right.im;
	return result;
}
__host__ __device__ complex complex_sub(complex left, complex right) {
	complex result;
	result.re = left.re - right.re;
	result.im = left.im - right.im;
	return result;
}
__host__ __device__ complex complex_mult(complex left, complex right) {
	complex result;
	result.re = left.re * right.re - left.im * right.im;
	result.im = left.re * right.im + left.im * right.re;
	return result;
}
__host__ void stampaj(complex x) {
	printf("%g+%gi ", x.re, x.im);
}
__host__ void stampajNiz(complex *A, int N) {
	int i;
	for (i = 0; i < N; i++) {
		stampaj(A[i]);
		putchar(' ');
		if (i && !(i % 7))
			putchar('\n');
	}
//	printf("kraj reda!\n");
}
__host__ void stampajMatricu(complex *m_A, int N) {
	int i;
	for (i = 0; i < N; i++) {
		stampajNiz(m_A + i * N, N);
	}
}
