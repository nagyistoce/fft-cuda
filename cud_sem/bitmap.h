////////////////////////////////////////////////////////
/*	                                                  */
/* Header file sa deklaracijama funkcija potrebnih za */
/* obradu .bmp fajlova, kao i transformacije	        */
/* dobijenih rezultata u zeljene za dalju obradu.     */
/*	                                                  */
/* CUDA C Programiranje                               */
/* Seminarski rad: 2D FFT	                           */
/* Ognjen Kocic 59/2010 Stefan Isidorovic 25/2010     */
/*                                                    */
////////////////////////////////////////////////////////


#include<stdio.h>
#include<stdlib.h>

#ifndef _bitmap_h
#define _bitmap_h

typedef struct{ /* RGB */
	unsigned char RGB[3];
}RGB;

/* Structure for defining INFOHEADER */
typedef struct{
	unsigned int sizeDib; /* Number of bytes in the DIB header (from this point) */
	unsigned int width; /* Width of the bitmap in pixels */
	unsigned int height; /* Height of the bitmap in pixels. Positive for bottom to top pixel order. Negative for top to bottom pixel order. */
	unsigned short int planes; /* Number of color planes being used */
	unsigned short int bpp; /* Number of bits per pixel */
	unsigned int xresolution; /* Horizontal resolution of the image */
	unsigned int yresolution; /* Vertical resolution of the image */
	unsigned int compression; /* BI_RGB, no pixel array compression used */
	unsigned int imagesize; /* Size of the raw data in the pixel array (including padding) */
	unsigned int colors; /* Number of colors in the palette */
	unsigned int impcolors; /* 0 means all colors are important */
}INFOHEADER;

/* Structure for defining header */
typedef struct{
	short int type; 	/* File type  Magic number (unsigned integer 66, 77) */
	unsigned int	size; /* Size of the BMP file */
	unsigned short int reserved1,reserved2; /* Application specific  */
	unsigned int	offset; 	/* Offset where the pixel array (bitmap data) can be found */
}HEADER;
	
typedef struct { /* BITMAP */
	int width; /* BITMAP width */
	int height; /* BITMAP height */
	FILE* f; /* File pointer to .bmp file */
	INFOHEADER info; /* INFOHEADER of BITMAP */
	HEADER head; /* HEADER of BITMAP */
	float** matrix; /* Pixel Matrix containing information about every pixel */
}BITMAP;

typedef struct {
	int dim; /* Dimension of filter matrix */
	float factor, bias; /* factor of multiplication and bias for brightness (add */
	float** matrix; /* Filter matrix */
}FILTER;

////////////////////////////////////////////////////////
/** Functions for reading and transforming .bmp file **/
////////////////////////////////////////////////////////

/* Reads INFOHEADER of .bmp file
 * INPUT: file pointer to .bmp file, 
 * RETURN: INFOHEADER structure containing informations
 */
INFOHEADER readInfo( FILE* f ); 

/* Reads HEADER of .bmp file
 * INPUT: file pointer to .bmp file
 * RETURN: HEADER structure containing informatios
 */
HEADER readHeader(FILE *f );

/* Print INFOHEADER on stdout
 * INFPUT: INFOHEADER structure
 * RETURN: void
 */
void printInfoheader(INFOHEADER info);

/* Print HEADER on stdout
 * INPUT: HEADER structure
 * RETURN: void
 */
void printHeader(HEADER head);

/* Loading pixel matrix from .bmp file and storing it in mat
 * INPUT: File pointer to .bmp, float matrix (which will contain results)
 * RETURN: void
 */
void loadImage(FILE* f, INFOHEADER info, float** mat);

/* Alocating memory for float matrix
 * INPUT: width of matrix, height of matrix
 * RETURN: Pointer to alocated memory
 */
float** createMatrix(int m_wid, int m_high);

/* Print matrix on stdout
 * INPUT: float matrix
 * RETURN void  
 */
void printMatrix(float** mat, int height, int width); 

/* Checking is file .bmp
 * INPUT: HEADER structure, INFOHEADER structure, path to .bmp 
 * RETURN: void
 */
void isBMP(HEADER head, INFOHEADER info,char* path); /* FLAG razmotriti konverziju iz 32bit*/

/* Writing new .bmp file
 * INPUT: BITMAP structure, path
 * RETURN: void
 */
void writeBMP( BITMAP pic, char* path);

/* Reading .bmp file and creating BITMAP structure with all necesary informations
 * INPUT: path to .bmp file
 * RETURN: BITMAP structure
 */
BITMAP createBMP(char* path);

/* Reads FILTER structure from file
 * INPUT: path to file
 * RETURN: FILTER structure
 */
FILTER readFilter(char* path);

////////////////////////////////////////////////////////
/**** Host functions (for oldschool convolution :) ****/
////////////////////////////////////////////////////////

/* Extracting original matrix from "bordered" matrix
 * INPUT: BITMAP structure (where will original matrix be stored), FILTER structure, "borderd" matrix 
 * RETURN: void
 */
void extractMatrix(BITMAP pic, FILTER fil, float** output);

/* Creating "bordered" matrix
 * INPUT: BITMAP structure, FILTER fil, output matrix which represent result of function
 * RETURN: void
 */
float** outputMatrix(BITMAP pic, FILTER fil);

/* Applying convolution on image
 * INPUT: width and height of output matrix, BITMAP structure, FILTER structure, output matrix
 * RETURN: matrix as result of applying convolution on output (with implemented "extraction")
 */
float** convolution(int width, int height, BITMAP pic, FILTER fil, float** output);

////////////////////////////////////////////////////////
/****   Function Necesary for FFT implementation   ****/
////////////////////////////////////////////////////////

/* Creating matrix pow2 x pow2 from input matrix (width and height are dimensions of input
 * INPUT: pow2, input matrix, and dimensions of input matrix (width and height)
 * RETURN: float** matrix dimension pow2 x pow2. With inserted zeros
 */
float** outputMatrixFFT(int pow2, float** input, int width, int height);/* width and height of input matrix */

/* Finding power of 2 closest to max of width and height
 * INPUT: width, height
 * RETURN: closeset power of 2
 */
int powOf2(int width, int height);

/* Transforming matrix to array (float** to float*)
 * INPUT: input matrix, dimension of matrix (matrix size n x n)
 * RETURN: resulting array
 */
float* transformToArray(float** input, int n);

/* Transforming array to matrix (float* to float**)
 * INPUT: input array, dimension of origin matrix
 * RETURN: resulting matrix
 */
float** transformToMatrix(float* input, int n);

/* Extracting submatrix direct into BITMAP structure
 * INPUT: BITMAP structure, input matrix
 * RETURN: void
 */
void extractFFTMatrix(BITMAP pic, float** input);

#endif