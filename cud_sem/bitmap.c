#include "bitmap.h"

/* Checking google group! prokletinja odbija saradnju za sad -.- */

INFOHEADER readInfo( FILE* f ){ /* Reading infoheader from .bmp file */
	INFOHEADER info;
	
	/* Number of bytes in the DIB header (from this point) */
	fseek(f,14,0);
	fread(&info.sizeDib,1,4,f);
	
	/* Width of the bitmap in pixels */
	fseek(f,18,0);
	fread(&info.width,1,4,f);
	width = info.width;
	
	/* Height of the bitmap in pixels. Positive for bottom to top pixel order. Negative for top to bottom pixel order. */
	fseek(f,22,0);
	fread(&info.height,1,4,f);
	height = info.height;
	
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

void writeInfoheader(INFOHEADER info){ /* Printing INFOHEADER on stdout */
	printf(" *** INFOHEADER *** \n");
	printf(" Size of DIB header: %d \n",info.sizeDib);
	printf(" Dimensions : %d x %d \n",info.width,info.height);
	printf(" Number of planes: %d \n",info.planes);
	printf(" Bits per pixel: %d \n",info.bpp);
	printf(" Compresion: %d \n",info.compression);
	printf(" Raw Size in bytes: %d \n",info.imagesize);
	printf(" Horizontal x Vertical resolution: %d x %d \n",info.xresolution,info.yresolution);
	printf(" Number of used colors: %d \n",info.colors);
	printf(" Number of important colors: %d \n",info.impcolors);
}

void writeHeader(HEADER head){ /* Printing HEADER on stdout */
	printf(" *** Header  *** \n");
	printf(" Type : %d \n",head.type);
	printf(" Size : %d \n",head.size);
	printf(" Reserved1 : %d \n Reserved2 : %d \n",head.reserved1, head.reserved2);
	printf(" Offset : %d \n",head.offset);
}

void loadImage(FILE* f, float** mat){ /* Loading and creating matrix from .bmp */
	int i,j;
	RGB temp;
	int rgb;
	long int pos = 51;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
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

float** createMatrix(void){ /* Allocating matrix */
	float** mat;
	int i;
	mat = (float**) malloc(sizeof( float* )*height);
	if(mat == NULL){
		printf(" Failed alocating rows ");
		exit(1);
	}
	for(i = 0; i < height; i++){
		mat[i] = ( float* ) malloc(sizeof(float)*width);
		if(mat[i] == NULL){
			printf(" Failed alocating columns %d ",i);
			exit(1);
		}
	}
	return mat;
}

void writeMatrix(float** mat){ /* Writing matrix on stdout */
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
        if(info.bpp != 24){
		printf("%s : The file is not 24 bit! \n",path);
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
	int height = pic.height;
	int width = pic.width;
	
	fseek(pic.f,0,0);
	fread(header,54,1,pic.f);

	fseek(out,0,0);
	fwrite(header,54,1,out);

	for(i = 0; i < height; i++){
		for(j = 0 ; j < width; j++){
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

BITMAP createBMP(char* path){
	BITMAP pic;
	pic.f = fopen(path,"rb");
	pic.info = readInfo(pic.f);
	pic.head = readHeader(pic.f);
	pic.height = height;
	pic.width = width;
	isBMP(pic.head,pic.info,path);
	pic.matrix = createMatrix();
	loadImage(pic.f,pic.matrix);
	return pic;
}

