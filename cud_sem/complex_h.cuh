#ifndef COMPLEX_H_CU
#define COMPLEX_H_CU
typedef struct complex_t {
	double re;
	double im;
} complex;
__host__ __device__ complex complex_from_polar(double r, double theta_radians);
__host__ __device__ complex complex_add(complex left, complex right);
__host__ __device__ complex complex_sub(complex left, complex right);
__host__ __device__ complex complex_mult(complex left, complex right);
__host__ void stampaj(complex x);
__host__ void stampajNiz(complex *A, int N);
__host__ void stampajMatricu(complex *m_A, int N);
#endif
