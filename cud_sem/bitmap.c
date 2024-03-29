
////////////////////////////////////////////////////////
/*	                                                  */
/* C file sa implementacijama funkcija potrebnih za   */
/* obradu .bmp fajlova, kao i transformacije	        */
/* dobijenih rezultata u zeljene za dalju obradu.     */
/*									                                  */
/* CUDA C Programiranje                               */
/* Seminarski rad: 2D FFT	                 				  */
/* Ognjen Kocic 59/2010 Stefan Isidorovic 25/2010     */
/* 									                                 */
////////////////////////////////////////////////////////

#include "bitmap.h"
#include<math.h>
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))

INFOHEADER readInfo( FILE* f ){ /* Reading infoheader from .bmp file */
	INFOHEADER info;
	
	/* Number of bytes in the DIB header (from this point) */
	fseek(f,14,0);
	fread(&info.sizeDib,1,4,f);
	
	/* Width of the bitmap in pixels */
	fseek(f,18,0);
	fread(&info.width,1,4,f);
	
	/* Height of the bitmap in pixels. Positive for bottom to top pixel order. Negative for top to bottom pixel order. */
	fseek(f,22,0);
	fread(&info.height,1,4,f);
	
	/* Number of color planes being used */
	fseek(f,0x1A,0);
	fread(&info.planes,1,2,f);
	
	/* Number of bits per pixel */
	fseek(f,0x1C,0);
	fread(&info.bpp,1,2,f);
	
	/* BI_RGB, no pixel array compression used  0 = Normal, 1 = 8 bpp, 2 = 4 bpp */
	fseek(f,0x1E,0);
	fread(&info.compression,1,4,f);
	
	/* Size of the raw data in the pixel array (including padding) */
	fseek(f,0x22,0);
	fread(&info.imagesize,1,4,f);
	
	/* Horizontal resolution of the image */
	fseek(f,0x26,0);
	fread(&info.xresolution,1,4,f);
	
	/* Vertical resolution of the image */
	fseek(f,0x2A,0);
	fread(&info.yresolution,1,4,f);
	
	/* Number of colors in the palette, 0 = full color set used */
	fseek(f,0x2E,0);
	fread(&info.colors,1,4,f);
	
	/* Number of important colors, 0 = all colors are important */
	fseek(f,0x32,0);
	fread(&info.impcolors,1,4,f);
	
	return info;
}

HEADER readHeader( FILE* f ){ /* Reading header from .bmp file */
	HEADER head;
	/*  File type, type, char */
	fseek(f,0,0);
      fread(&head.type,1,2,f);
	/* Size, size, unsigned int */
	fseek(f,2,0);
 	fread(&head.size,1,4,f);
	/* Reserved1, Reserved2, unsigned short int */
	fseek(f,4,0);
	fread(&head.reserved1,1,2,f);
	fseek(f,6,0);
	fread(&head.reserved2,1,2,f);
	/* Offset, unsigned int */
	fseek(f,10,0);
	fread(&head.type,1,4,f);
	return head;
}

void printInfoheader(INFOHEADER info){ /* Printing INFOHEADER on stdout */
	printf(" *** INFOHEADER *** \n");
	printf(" Size of DIB header: %d \n",info.sizeDib);
	printf(" Dimensions : %d x %d \n",info.width,info.height);
	printf(" Number of planes: %hd \n",info.planes);
	printf(" Bits per pixel: %hd \n",info.bpp);
	printf(" Compresion: %d \n",info.compression);
	printf(" Raw Size in bytes: %d \n",info.imagesize);
	printf(" Horizontal x Vertical resolution: %d x %d \n",info.xresolution,info.yresolution);
	printf(" Number of used colors: %d \n",info.colors);
	printf(" Number of important colors: %d \n",info.impcolors);
}

void printHeader(HEADER head){ /* Printing HEADER on stdout */
	printf(" *** Header  *** \n");
	printf(" Type : %hd \n",head.type);
	printf(" Size : %d \n",head.size);
	printf(" Reserved1 : %hd \n Reserved2 : %hd \n",head.reserved1, head.reserved2);
	printf(" Offset : %d \n",head.offset);
}

void loadImage(FILE* f, INFOHEADER info, float** mat){ /* Loading and creating matrix from .bmp */
	int i,j;
	RGB temp;
	int rgb;
	long int pos = 51;
	for (i = 0; i < info.height; i++){
		for (j = 0; j < info.width; j++){
			pos += 3;
			fseek(f, pos, 0);
			fread(&temp, sizeof(RGB), 1,f);
			rgb = temp.RGB[0];
			rgb = (rgb << 8) + temp.RGB[1];
			rgb = (rgb << 8) + temp.RGB[2];
			mat[i][j] = (float) rgb;
                }
        }
}

float** createMatrix(int m_wid, int m_high){ /* Allocating matrix */
	float** mat;
	int i;
	mat = (float**) malloc(sizeof( float* )*m_high);
	if(mat == NULL){
		printf(" Failed alocating rows ");
		exit(1);
	}
	for(i = 0; i < m_high; i++){
		mat[i] = ( float* ) malloc(sizeof(float)*m_wid);
		if(mat[i] == NULL){
			printf(" Failed alocating columns %d ",i);
			exit(1);
		}
	}
	return mat;
}

void printMatrix(float** mat, int height, int width){ /* Printing matrix on stdout */
	int i,j;
	
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			printf(" %.1f ",mat[i][j]);
		}
		printf("\n");
	}
}

void isBMP(HEADER head, INFOHEADER info,char* path){/* Checking is file bmp and reading header */  
        if (head.type == 66 || head.type == 77) {
                printf("%s : The file is not BMP format! \n",path);
                exit(0);
        }
        /* FLAG */
        if(info.bpp > 24){
		printf("%s : The file compression is greater than 24 bit! \n",path);
		exit(1);
        }
}

void writeBMP( BITMAP pic, char* opath){ /* Creating new .bmp file */
	FILE* out = fopen(opath,"wb");
	int i,j;
	RGB temp;
	long int pos = 51;
	char header[54];
	int pom;
	
	fseek(pic.f,0,0);
	fread(header,54,1,pic.f);
	fseek(out,0,0);
	fwrite(header,54,1,out);
	
	for(i = 0; i < pic.height; i++){
		for(j = 0 ; j < pic.width; j++){
			pos += 3;
			fseek(out,pos,0);
			/* Possible future problem with explicit converting from float to int! */
			pom =(int) pic.matrix[i][j];
			temp.RGB[2] = pom & 0x000000FF;
			pom = pom >> 8;
			temp.RGB[1] = pom & 0x000000FF;
			pom = pom >> 8;
			temp.RGB[0] = pom & 0x000000FF;
			fwrite(&temp,(sizeof(RGB)),1,out);
		}
	}
	fclose(out);
}

BITMAP createBMP(char* path){ /* Constructor for BMP structure */
	BITMAP pic;
	pic.f = fopen(path,"rb");
	pic.info = readInfo(pic.f);
	pic.head = readHeader(pic.f);
	pic.height = pic.info.height;
	pic.width = pic.info.width;
	isBMP(pic.head,pic.info,path);
	pic.matrix = createMatrix(pic.width, pic.height);
	loadImage(pic.f,pic.info,pic.matrix);
	return pic;
}

float** outputMatrix(BITMAP pic, FILTER fil){ /* Creating matrix with border for convolution */
	int out_width = (pic.width + 2 * (fil.dim/2)), out_height = (pic.height+(fil.dim/2)*2);
	float** output = createMatrix(out_width, out_height);
	int i, j, k, ix = fil.dim/2, iy = 0;
	
	for(i = 0;i < pic.height; i++){
		iy = fil.dim/2;
		for(j = 0; j < pic.width; j++){
			output[ix][iy] = pic.matrix[i][j];
			iy += 1;
			
		}
		ix += 1;
	}
	for(i = 0;i < fil.dim/2; i++){
		for(j = 0;j < out_width; j++){
			output[i][j] = 0;
			output[out_height - i - 1][j] = 0;
		}
		for(k = 0;k < out_height; k++){
			output[k][i] = 0;
			output[k][out_width - i - 1] = 0;
		}
	}
	return output;
}

void extractMatrix(BITMAP pic, FILTER fil, float** output){ /* Extracting bmp pixel matrix from output matrix */
	
	int i, j, ix = 0, iy = 0, grw = pic.width + fil.dim/2, grh = pic.height + fil.dim/2;
	for(i = fil.dim/2;i < grh; i++){
		for(j = fil.dim/2; j < grw; j++){
			pic.matrix[ix][iy] = output[i][j];
			iy += 1;
		}
		ix += 1;
	}
}

float** convolution(int width, int height,BITMAP pic, FILTER fil, float** output){ /* width and height of  output matrix,Applying convolution on matrix  */
	/* input: width and heigth of output matrix, BITMAP structure, filter, output matrix, factor, bias */
	float** result = createMatrix(pic.width,pic.height);
	
	int i, j, k, l; 
	int r,g,b;
	int imx, imy, temp;
	for(i = fil.dim / 2; i < pic.height + fil.dim/2; i++){
		for(j = fil.dim / 2; j < pic.width + fil.dim / 2; j++){
			r = 0;
			g = 0;
			b = 0;
			for(k = 0; k < fil.dim; k++)
				for(l = 0; l < fil.dim; l++){
					imx = i - fil.dim/2 + k;
					imy = j - fil.dim/2 + l;
					temp = (int) output[imx][imy];
					b += (temp & 0x000000FF)*fil.matrix[k][l];
					g += ((temp >> 8) & 0x000000FF)*fil.matrix[k][l];
					r += ((temp >> 16) & 0x000000FF)*fil.matrix[k][l];
				}
			result[i - fil.dim/2][j - fil.dim/2] = (float) ( ( MIN(MAX((int)(fil.factor * r + fil.bias), 0), 255) << 16 ) + ( MIN(MAX((int)(fil.factor * g + fil.bias), 0), 255) << 8 ) + ( MIN(MAX((int)(fil.factor * b + fil.bias), 0), 255) )); 		
		}
	}
	
	return result;
}

float** outputMatrixFFT(int pow2, float** input, int width, int height){ /* pow2, input matrix, dimensions of input matrix */
	int i, j;
	float** output = createMatrix(pow2,pow2);
	for(i = 0; i < pow2; i++)
		for(j = 0; j < pow2; j++){
			if( i < height && j < width)
				output[i][j] = input[i][j];
			else
				output[i][j] = 0; /* FLAG */
		}
	return output;
}

int powOf2(int width, int height){
	int pow2 = 1;
	int max = MAX(width, height);
	while(max > pow2)
		pow2 = pow2 << 1;
	return pow2;
}

FILTER readFilter(char* path){
	FILTER output;
	FILE* f = fopen(path,"r");
	int count = 0;
	int i, j;
	float temp;
	while(fscanf(f,"%f ",&temp) == 1)
		count++;
	output.dim = (int) sqrt(count - 2);
	output.matrix = createMatrix(output.dim,output.dim);
	fclose(f);
	f = fopen(path, "r");
	for(i = 0; i < output.dim; i++)
		for(j = 0; j < output.dim; j++)
			fscanf(f,"%f ",&output.matrix[i][j]);
	fscanf(f,"%f ",&output.factor);
	fscanf(f,"%f ",&output.bias);
	fclose(f);
	return output;
}

float* transformToArray(float** input, int n){ /* n is dimension of matrix! */
	float* output =  (float*) malloc(n*n*sizeof(float));
	int i, j, array = 0;
	if(output == NULL){
		printf(" Failed alocating array ");
		exit(1);
	}
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++){
			output[array] = input[i][j];
			array += 1;
		}
	return output;
}

float** transformToMatrix(float* input,int n){
	float** output = createMatrix(n,n);
	int i=0,x=0,y=0;
	output[x][y] = input[i];
	y += 1;
	for(i = 1; i < n*n; i++){
		output[x][y] = input[i];
		if( (1 + i) % n == 0 ){
			x += 1;
			y = 0;
		}
		else
			y += 1;
	}
	
	return output;
}

void extractFFTMatrix(BITMAP pic, float** input){
	int i,j;
	for(i = 0; i < pic.height; i++)
		for(j = 0; j < pic.width; j++)
			pic.matrix[i][j] = input[i][j];
}

////////////////////////////////////////////////////////
/*									  */
/* C file sa implementacijama funkcija potrebnih za   */
/* obradu .bmp fajlova, kao i transformacije	        */
/* dobijenih rezultata u zeljene za dalju obradu.     */
/*									  */
/* CUDA C Programiranje                               */
/* Seminarski rad: 2D FFT					  */
/* Ognjen Kocic 59/2010 Stefan Isidorovic 25/2010     */
/* 									  */
////////////////////////////////////////////////////////

#include "bitmap.h"
#include<math.h>
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))

INFOHEADER readInfo( FILE* f ){ /* Reading infoheader from .bmp file */
	INFOHEADER info;
	
	/* Number of bytes in the DIB header (from this point) */
	fseek(f,14,0);
	fread(&info.sizeDib,1,4,f);
	
	/* Width of the bitmap in pixels */
	fseek(f,18,0);
	fread(&info.width,1,4,f);
	
	/* Height of the bitmap in pixels. Positive for bottom to top pixel order. Negative for top to bottom pixel order. */
	fseek(f,22,0);
	fread(&info.height,1,4,f);
	
	/* Number of color planes being used */
	fseek(f,0x1A,0);
	fread(&info.planes,1,2,f);
	
	/* Number of bits per pixel */
	fseek(f,0x1C,0);
	fread(&info.bpp,1,2,f);
	
	/* BI_RGB, no pixel array compression used  0 = Normal, 1 = 8 bpp, 2 = 4 bpp */
	fseek(f,0x1E,0);
	fread(&info.compression,1,4,f);
	
	/* Size of the raw data in the pixel array (including padding) */
	fseek(f,0x22,0);
	fread(&info.imagesize,1,4,f);
	
	/* Horizontal resolution of the image */
	fseek(f,0x26,0);
	fread(&info.xresolution,1,4,f);
	
	/* Vertical resolution of the image */
	fseek(f,0x2A,0);
	fread(&info.yresolution,1,4,f);
	
	/* Number of colors in the palette, 0 = full color set used */
	fseek(f,0x2E,0);
	fread(&info.colors,1,4,f);
	
	/* Number of important colors, 0 = all colors are important */
	fseek(f,0x32,0);
	fread(&info.impcolors,1,4,f);
	
	return info;
}

HEADER readHeader( FILE* f ){ /* Reading header from .bmp file */
	HEADER head;
	/*  File type, type, char */
	fseek(f,0,0);
      fread(&head.type,1,2,f);
	/* Size, size, unsigned int */
	fseek(f,2,0);
 	fread(&head.size,1,4,f);
	/* Reserved1, Reserved2, unsigned short int */
	fseek(f,4,0);
	fread(&head.reserved1,1,2,f);
	fseek(f,6,0);
	fread(&head.reserved2,1,2,f);
	/* Offset, unsigned int */
	fseek(f,10,0);
	fread(&head.type,1,4,f);
	return head;
}

void printInfoheader(INFOHEADER info){ /* Printing INFOHEADER on stdout */
	printf(" *** INFOHEADER *** \n");
	printf(" Size of DIB header: %d \n",info.sizeDib);
	printf(" Dimensions : %d x %d \n",info.width,info.height);
	printf(" Number of planes: %hd \n",info.planes);
	printf(" Bits per pixel: %hd \n",info.bpp);
	printf(" Compresion: %d \n",info.compression);
	printf(" Raw Size in bytes: %d \n",info.imagesize);
	printf(" Horizontal x Vertical resolution: %d x %d \n",info.xresolution,info.yresolution);
	printf(" Number of used colors: %d \n",info.colors);
	printf(" Number of important colors: %d \n",info.impcolors);
}

void printHeader(HEADER head){ /* Printing HEADER on stdout */
	printf(" *** Header  *** \n");
	printf(" Type : %hd \n",head.type);
	printf(" Size : %d \n",head.size);
	printf(" Reserved1 : %hd \n Reserved2 : %hd \n",head.reserved1, head.reserved2);
	printf(" Offset : %d \n",head.offset);
}

void loadImage(FILE* f, INFOHEADER info, float** mat){ /* Loading and creating matrix from .bmp */
	int i,j;
	RGB temp;
	int rgb;
	long int pos = 51;
	for (i = 0; i < info.height; i++){
		for (j = 0; j < info.width; j++){
			pos += 3;
			fseek(f, pos, 0);
			fread(&temp, sizeof(RGB), 1,f);
			rgb = temp.RGB[0];
			rgb = (rgb << 8) + temp.RGB[1];
			rgb = (rgb << 8) + temp.RGB[2];
			mat[i][j] = (float) rgb;
                }
        }
}

float** createMatrix(int m_wid, int m_high){ /* Allocating matrix */
	float** mat;
	int i;
	mat = (float**) malloc(sizeof( float* )*m_high);
	if(mat == NULL){
		printf(" Failed alocating rows ");
		exit(1);
	}
	for(i = 0; i < m_high; i++){
		mat[i] = ( float* ) malloc(sizeof(float)*m_wid);
		if(mat[i] == NULL){
			printf(" Failed alocating columns %d ",i);
			exit(1);
		}
	}
	return mat;
}

void printMatrix(float** mat, int height, int width){ /* Printing matrix on stdout */
	int i,j;
	
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			printf(" %.1f ",mat[i][j]);
		}
		printf("\n");
	}
}

void isBMP(HEADER head, INFOHEADER info,char* path){/* Checking is file bmp and reading header */  
        if (head.type == 66 || head.type == 77) {
                printf("%s : The file is not BMP format! \n",path);
                exit(0);
        }
        /* FLAG */
        if(info.bpp > 24){
		printf("%s : The file compression is greater than 24 bit! \n",path);
		exit(1);
        }
}

void writeBMP( BITMAP pic, char* opath){ /* Creating new .bmp file */
	FILE* out = fopen(opath,"wb");
	int i,j;
	RGB temp;
	long int pos = 51;
	char header[54];
	int pom;
	
	fseek(pic.f,0,0);
	fread(header,54,1,pic.f);
	fseek(out,0,0);
	fwrite(header,54,1,out);
	
	for(i = 0; i < pic.height; i++){
		for(j = 0 ; j < pic.width; j++){
			pos += 3;
			fseek(out,pos,0);
			/* Possible future problem with explicit converting from float to int! */
			pom =(int) pic.matrix[i][j];
			temp.RGB[2] = pom & 0x000000FF;
			pom = pom >> 8;
			temp.RGB[1] = pom & 0x000000FF;
			pom = pom >> 8;
			temp.RGB[0] = pom & 0x000000FF;
			fwrite(&temp,(sizeof(RGB)),1,out);
		}
	}
	fclose(out);
}

BITMAP createBMP(char* path){ /* Constructor for BMP structure */
	BITMAP pic;
	pic.f = fopen(path,"rb");
	pic.info = readInfo(pic.f);
	pic.head = readHeader(pic.f);
	pic.height = pic.info.height;
	pic.width = pic.info.width;
	isBMP(pic.head,pic.info,path);
	pic.matrix = createMatrix(pic.width, pic.height);
	loadImage(pic.f,pic.info,pic.matrix);
	return pic;
}

float** outputMatrix(BITMAP pic, FILTER fil){ /* Creating matrix with border for convolution */
	int out_width = (pic.width + 2 * (fil.dim/2)), out_height = (pic.height+(fil.dim/2)*2);
	float** output = createMatrix(out_width, out_height);
	int i, j, k, ix = fil.dim/2, iy = 0;
	
	for(i = 0;i < pic.height; i++){
		iy = fil.dim/2;
		for(j = 0; j < pic.width; j++){
			output[ix][iy] = pic.matrix[i][j];
			iy += 1;
			
		}
		ix += 1;
	}
	for(i = 0;i < fil.dim/2; i++){
		for(j = 0;j < out_width; j++){
			output[i][j] = 0;
			output[out_height - i - 1][j] = 0;
		}
		for(k = 0;k < out_height; k++){
			output[k][i] = 0;
			output[k][out_width - i - 1] = 0;
		}
	}
	return output;
}

void extractMatrix(BITMAP pic, FILTER fil, float** output){ /* Extracting bmp pixel matrix from output matrix */
	
	int i, j, ix = 0, iy = 0, grw = pic.width + fil.dim/2, grh = pic.height + fil.dim/2;
	for(i = fil.dim/2;i < grh; i++){
		for(j = fil.dim/2; j < grw; j++){
			pic.matrix[ix][iy] = output[i][j];
			iy += 1;
		}
		ix += 1;
	}
}

float** convolution(int width, int height,BITMAP pic, FILTER fil, float** output){ /* width and height of  output matrix,Applying convolution on matrix  */
	/* input: width and heigth of output matrix, BITMAP structure, filter, output matrix, factor, bias */
	float** result = createMatrix(pic.width,pic.height);
	
	int i, j, k, l; 
	int r,g,b;
	int imx, imy, temp;
	for(i = fil.dim / 2; i < pic.height + fil.dim/2; i++){
		for(j = fil.dim / 2; j < pic.width + fil.dim / 2; j++){
			r = 0;
			g = 0;
			b = 0;
			for(k = 0; k < fil.dim; k++)
				for(l = 0; l < fil.dim; l++){
					imx = i - fil.dim/2 + k;
					imy = j - fil.dim/2 + l;
					temp = (int) output[imx][imy];
					b += (temp & 0x000000FF)*fil.matrix[k][l];
					g += ((temp >> 8) & 0x000000FF)*fil.matrix[k][l];
					r += ((temp >> 16) & 0x000000FF)*fil.matrix[k][l];
				}
			result[i - fil.dim/2][j - fil.dim/2] = (float) ( ( MIN(MAX((int)(fil.factor * r + fil.bias), 0), 255) << 16 ) + ( MIN(MAX((int)(fil.factor * g + fil.bias), 0), 255) << 8 ) + ( MIN(MAX((int)(fil.factor * b + fil.bias), 0), 255) )); 		
		}
	}
	
	return result;
}

float** outputMatrixFFT(int pow2, float** input, int width, int height){ /* pow2, input matrix, dimensions of input matrix */
	int i, j;
	float** output = createMatrix(pow2,pow2);
	for(i = 0; i < pow2; i++)
		for(j = 0; j < pow2; j++){
			if( i < height && j < width)
				output[i][j] = input[i][j];
			else
				output[i][j] = 0; /* FLAG */
		}
	return output;
}

int powOf2(int width, int height){
	int pow2 = 1;
	int max = MAX(width, height);
	while(max > pow2)
		pow2 = pow2 << 1;
	return pow2;
}

FILTER readFilter(char* path){
	FILTER output;
	FILE* f = fopen(path,"r");
	int count = 0;
	int i, j;
	float temp;
	while(fscanf(f,"%f ",&temp) == 1)
		count++;
	output.dim = (int) sqrt(count - 2);
	output.matrix = createMatrix(output.dim,output.dim);
	fclose(f);
	f = fopen(path, "r");
	for(i = 0; i < output.dim; i++)
		for(j = 0; j < output.dim; j++)
			fscanf(f,"%f ",&output.matrix[i][j]);
	fscanf(f,"%f ",&output.factor);
	fscanf(f,"%f ",&output.bias);
	fclose(f);
	return output;
}

float* transformToArray(float** input, int n){ /* n is dimension of matrix! */
	float* output =  (float*) malloc(n*n*sizeof(float));
	int i, j, array = 0;
	if(output == NULL){
		printf(" Failed alocating array ");
		exit(1);
	}
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++){
			output[array] = input[i][j];
			array += 1;
		}
	return output;
}

float** transformToMatrix(float* input,int n){
	float** output = createMatrix(n,n);
	int i=0,x=0,y=0;
	output[x][y] = input[i];
	y += 1;
	for(i = 1; i < n*n; i++){
		output[x][y] = input[i];
		if( (1 + i) % n == 0 ){
			x += 1;
			y = 0;
		}
		else
			y += 1;
	}
	
	return output;
}

void extractFFTMatrix(BITMAP pic, float** input){
	int i,j;
	for(i = 0; i < pic.height; i++)
		for(j = 0; j < pic.width; j++)
			pic.matrix[i][j] = input[i][j];
}
