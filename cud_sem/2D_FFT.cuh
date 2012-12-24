#ifndef D_FTT_CUH
#define D_FTT_CUH

#define PI 3.1415926535897932
#define EPSILON .0001
#define SWAP_COMPLEX(A, i, j, tmp) tmp.re = A[i].re;tmp.im = A[i].im;A[i].re = A[j].re;A[i].im = A[j].im;A[j].re = tmp.re;A[j].im = tmp.im

#include "complex_h.cuh"

__global__ void d_matrixTranspose(complex * d_m_A, int N);
__global__ void d_1D_FFT(complex* d_m_A, int N, int ln);
__global__ void d_bitreversal(complex *d_m_A,int N);
__host__ __device__ int revert(int x, int N);
__host__ void obradaHostMatrice(complex *h_m_A, int N);
__host__ void checkCUDAError(const char *msg);
#endif
