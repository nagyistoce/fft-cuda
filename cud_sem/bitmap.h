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
	int width; 
	int height;
	FILE* f;
	INFOHEADER info;
	HEADER head;
	float** matrix;
}BITMAP;

int width,height;

INFOHEADER readInfo( FILE* f );
HEADER readHeader(FILE *f );
void writeInfoheader(INFOHEADER info);
void writeHeader(HEADER head);
void loadImage(FILE* f, float** mat);
float** createMatrix(void);
void writeMatrix(float** mat);
void isBMP(HEADER head, INFOHEADER info,char* path);
void writeBMP( BITMAP pic, char* path);
BITMAP createBMP(char* path);

#endif
