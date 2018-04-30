/////////////////////////////////////////////////////////////////////////////////
// This code belongs to "Energy Efficient Deep Learning” project group at      //
// Cornell University, Graduate School of Electrical and Computer Engineering. //
// This code is strictly for research and education. Do not use it for any     //
// commercial purposes.                                                        //
// Author: Jonathan Wu; Guangwei Chen                                          //
/////////////////////////////////////////////////////////////////////////////////
#ifndef LOADBMP_H
#define LOADBMP_H

#include <stdio.h>

typedef struct {
   unsigned short int type;                 /* Image type identifier   */
   unsigned int size;                       /* File size in bytes  */
   unsigned short int reserved1;
   unsigned short int reserved2;
   unsigned int offset;                     /* Offset to image data, bytes */
} HEADER;


typedef struct {
   unsigned int size;               /* Header size in bytes      */
   int width;
   int height;                /* Width and height of image */
   unsigned short int planes;       /* Number of colour planes   */
   unsigned short int bits;         /* Bits per pixel            */
   unsigned int compression;        /* Compression type          */
   unsigned int imagesize;          /* Image size in bytes       */
   int xresolution;
   int yresolution;     /* Pixels per meter          */
   unsigned int ncolours;           /* Number of colours         */
   unsigned int importantcolours;   /* Important colours         */
} INFOHEADER;

typedef struct {
   unsigned char r,g,b,junk;
} COLOURINDEX;

int load_bmp(int argc, char **argv, unsigned char b[][224], unsigned char g[][224], unsigned char r[][224]);
int ReadUShort(FILE *,unsigned short *,int);
int ReadUInt(FILE *,unsigned int *,int);


#endif
