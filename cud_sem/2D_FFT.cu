#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "complex_h.cuh"
#include "2D_FFT.cuh"

__host__ void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s.\nMesto greske: %s.\n", msg,
				cudaGetErrorString(err));
		exit(-1);
	}
}

/*Ovo je remek-delo! Od dimenzije niza 8 do dimenzije niza 1024 vraca obrnuti indeks u O(1):*/
__host__ __device__ int d_fast_revert(int x, int N) {
	/*Ovo ne vredi nesto preterano citati jedino da se ispisu maske na papiru tako ce biti jasno.*/
	switch (N) {
	case 1024:
		x = ((x & 0x0000001f) << 5) | ((x & 0x000003e0) >> 5);
		x = ((x & 0x00000063) << 3) | ((x & 0x00000318) >> 3)
				| (x & 0x00000084);
		x = ((x & 0x00000129) << 1) | ((x & 0x00000252) >> 1)
				| (x & 0x00000084);
		return x;
	case 512:
		int y = (x & 16);
		x = ((x & 0x0000000f) << 5) | ((x & 0x00000017) >> 5);
		x = ((x & 0x00000063) << 2) | ((x & 0x0000018c) >> 2);
		x = ((x & 0x000000a5) << 1) | ((x & 0x0000014a) >> 1) | y;
		return x;
	case 256:
		x = ((x & 0x0000000f) << 4) | ((x & 0x000000f0) >> 4);
		x = ((x & 0x00000033) << 2) | ((x & 0x000000cc) >> 2);
		x = ((x & 0x00000055) << 1) | ((x & 0x000000aa) >> 1);
		return x;

	case 128:
		y = x & 8;
		int z1 = (x & 0x00000002) << 4;
		int z2 = (x & 0x00000020) >> 4;
		x = ((x & 0x00000007) << 4) | ((x & 0x00000070) >> 4);
		x = ((x & 0x00000011) << 2) | ((x & 0x00000044) >> 2) | y | z1 | z2;
		return x;
	case 64:
		int reverted64[] = { 0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12,
				44, 28, 60, 2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14,
				46, 30, 62, 1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45,
				29, 61, 3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47,
				31, 63 };
		return reverted64[x];
	case 32:
		int reverted32[] = { 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22,
				14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15,
				31 };
		return reverted32[x];
	case 16:
		int reverted16[] = { 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7,
				15 };
		return reverted16[x];
	case 8:
		int reverted8[] = { 0, 4, 2, 6, 1, 5, 3, 7 };
		return reverted8[x];
	}
}

/*Kernel funkcija koja vrsi bitreversal za svaki red matrice na device-u:*/
__global__ void d_bitreversal(complex *d_m_A, int N) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	extern __shared__ complex A[];
	int i;
	/*Kopiramo odredjeni red matrice u odredjeni blok (iz globalne u shared memoriju coalesced kopiranje):*/
	A[tx] = d_m_A[N * bx + tx];
	__syncthreads();
	/*Menjamo elemente koji treba da se zamene (zbog if-a dolazi do male divergencije tredova jedan warp se izvrsava iz dva puta):*/
	int revertedIndex = d_fast_revert(tx, N);
	if (revertedIndex > tx) {
		complex tmp;
		SWAP_COMPLEX(A, tx, revertedIndex, tmp);
	}
	__syncthreads();

	/*Kopiramo rezultat u globalnu memoriju i to svaki blok na svoje mesto (coalesced kopiranje):*/
	d_m_A[bx * N + tx] = A[tx];
}

/*Kernel f-ja koja racuna jednodimenzioni fft za svaki red. Svaki blok adresira jedan red matrice.*/
__global__ void d_1D_FFT(complex* d_m_A, int N, int ln) {
	/*Deklarisemo niz nad kojim ce blok raditi 1d fft i u koji cemo prekopirati podatke iz globalne memorije.*/
	extern __shared__ complex A[];

	/*Indeks u matrici u globalnoj memoriji.*/
	int tx = threadIdx.x;
	int d_m_index = blockIdx.x * N + tx;
	int i, m;
	complex w;
	int trenutniIndeks;
	complex t, u;

	/*Vrsimo "coalesced" kopiranje.*/
	A[tx] = d_m_A[d_m_index];
	A[tx + N / 2] = d_m_A[d_m_index + N / 2];
	__syncthreads();

	/*Glavna petlja koja ide po nivoima sve ostalo regulisu tredovi.*/
	for (i = 1; i <= ln; i++) {
		m = 1 << i;
		w = complex_from_polar(1.0, 2. * PI / m);
		trenutniIndeks = (tx * m) % (N - 1);
		if (trenutniIndeks % m == 0) {
			w.re = 1;
			w.im = 0;
		} else {
			float degree = (2.0 * PI * (trenutniIndeks % m)) / m;
			w = complex_from_polar(1, degree);
		}
		t = complex_mult(w, A[trenutniIndeks + m / 2]);
		u = A[trenutniIndeks];
		A[trenutniIndeks] = complex_add(u, t);
		A[trenutniIndeks + m / 2] = complex_sub(u, t);
		__syncthreads();
	}

	/*vrsimo kopiranje natrag u glavnu memoriju:*/
	d_m_A[d_m_index] = A[tx];
	d_m_A[d_m_index + N / 2] = A[tx + N / 2];
}

/*Funkcija koja transponuje kvadratnu matricu:*/
__global__ void d_matrixTranspose(complex * d_m_A, int N) {
	extern __shared__ complex A[];
	int bdx = blockIdx.x;
	int tx = threadIdx.x;
	/*Kopiramo jedan red originalne matrice u SHARED memoriju:*/
	A[tx] = d_m_A[bdx * N + tx];
	__syncthreads();

	/*Vracamo u kolone pomocne matrice:*/
	d_m_A[tx * N + bdx] = A[tx];
	__syncthreads();

	/*Vracanje u pocetnu matricu:*/
}

/*Funkcija prima matricu slike sa hosta i radi 2D FFT na njoj:*/
__host__ void obradaHostMatrice(complex *h_m_A, int N) {
	/*Ako je N > 1024, ili N nije 2^k, matrica je prevelika za trenutnu obradu,
	 * ili nije formatirana, pa izlazimo iz programa.*/
	if ((N > 1024) || (N & (N - 1))) {
		fprintf(stderr,
				"GRESKA! Velicina matrice: %i. Zahtevi su da matrica mora biti stepen dvojke manji do jednak od 1024!\n");
		exit(EXIT_FAILURE);
	}

	/*Inicijalizujemo resurse i kopiramo host matricu na device:*/
	complex *d_m_A;
	cudaMalloc((void **) &d_m_A, N * N * sizeof(complex));
	checkCUDAError(
			"alokacija memorije za matricu na device-u, red 158 fajl 2D_FFT.cu ");
	cudaMemcpy(d_m_A, h_m_A, N * N * sizeof(complex), cudaMemcpyHostToDevice);
	checkCUDAError(
			"kopiranje matrice za obradu sa host-a na device, red 161 fajl 2D_FFT.cu ");

	/*Vrsimo obranje bitova za svaki red matrice posebno, a sve to jednom kernel funkcijom:*/
	d_bitreversal<<<N, N, N * sizeof(complex)>>>(d_m_A, N);
	cudaDeviceSynchronize();

	/*Primenjujemo 2D FFT za svaki red matrice posebno, a sve to jednom kernel funkcijom:*/
	int ln = (int) log2((double) N);
	d_1D_FFT<<<N, N / 2, N * sizeof(complex)>>>(d_m_A, N, ln);
	cudaDeviceSynchronize();

	/*Transponujemo matricu na device-u i okrecemo bitove svakom redu tako transponovane matrice
	 * da bi se pripremili za poziv 2D FFT za redove te matrice, tj. lepse receno za poziv
	 * 2D FFT-a za kolone pocetne matrice:*/
	d_matrixTranspose<<<N, N, N * sizeof(complex)>>>(d_m_A, N);
	cudaDeviceSynchronize();
	d_bitreversal<<<N, N, N * sizeof(complex)>>>(d_m_A, N);
	cudaDeviceSynchronize();

	/*Konacno vrsimo 2D FFT za kolone pocetne matrice i time zavrsavamo:*/
	d_1D_FFT<<<N, N / 2, N * sizeof(complex)>>>(d_m_A, N, ln);
	cudaDeviceSynchronize();
}
