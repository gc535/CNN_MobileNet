#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include "loadbmp.h"

using namespace std;


int load_bmp(int argc, char **argv, unsigned char b[][224], unsigned char g[][224], unsigned char r[][224]){
//int load_bmp(int argc, char **argv, float input[MAX_FMAP_SIZE]){


    HEADER fileHeader;
    INFOHEADER infoHeader;

    FILE*f = fopen(argv[1], "r");
    //fread(&fileHeader,sizeof(HEADER),1,f);
    //fread(&infoHeader,sizeof(INFOHEADER),1,f);
    
    // reading headers 
    ReadUShort(f,&fileHeader.type,0);
    fprintf(stderr,"Image type is: %X, should be 4D42\n",fileHeader.type);
    ReadUInt(f,&fileHeader.size,0);
    fprintf(stderr,"File size is %d bytes\n",fileHeader.size);
    ReadUShort(f,&fileHeader.reserved1,0);
    ReadUShort(f,&fileHeader.reserved2,0);
    ReadUInt(f,&fileHeader.offset,0);
    //fprintf(stderr,"Offset to image data is %d bytes\n",fileHeader.offset);
       
    if (fread(&infoHeader,sizeof(INFOHEADER),1,f) != 1) {
      fprintf(stderr,"Failed to read BMP info header\n\n");
      exit(-1);
    }
    
    //fprintf(stderr,"header size = %d, ",infoHeader.size);
    fprintf(stderr,"Image size = %d x %d\n",infoHeader.width,infoHeader.height);
    fprintf(stderr,"Number of colour planes is %d, ",infoHeader.planes);
    fprintf(stderr,"Bits per pixel is %d, ",infoHeader.bits);
    fprintf(stderr,"Compression type is %d\n",infoHeader.compression);
    fprintf(stderr,"image size = %d\n",infoHeader.imagesize);
    fprintf(stderr,"x * y (resolution) = %d x %d\n",infoHeader.xresolution, infoHeader.yresolution);
    fprintf(stderr,"Number of colours is %d, ",infoHeader.ncolours);
    fprintf(stderr,"Number of required colours is %d\n",infoHeader.importantcolours);
    

    // reading image data
    // unsigned char b[infoHeader.height][infoHeader.width];
    // unsigned char g[infoHeader.height][infoHeader.width];
    // unsigned char r[infoHeader.height][infoHeader.width];
    
    for(int i=0; i < infoHeader.height; i++){
        for(int j=0; j < infoHeader.width; j++){
            fread(&b[i][j], sizeof(unsigned char),1,f);
            fread(&g[i][j], sizeof(unsigned char),1,f);
            fread(&r[i][j], sizeof(unsigned char),1,f);
        }
    }
            
            
    //set the file name
    int delim;
    delim = 0;
    string file;
    file = argv[1];
    while(file.at(delim) != '.') delim++;  
    file.resize(delim); 
    
    //open the output file
    string suffix;
    suffix = ".txt";
    string output_file;
    output_file = file+suffix;
    cout << output_file <<endl;
    ofstream myfile (output_file.c_str());
    
    
    //save image data
    if (myfile.is_open())
    {
        myfile << "Red data:\n";
        for(int i=0; i < infoHeader.height; i++){
            for(int j=0; j < infoHeader.width; j++) myfile << (int)r[i][j] << " ";
             myfile << "\n";
        }
        myfile << "\n";
        myfile << "Green data:\n";
        for(int i=0; i < infoHeader.height; i++){
            for(int j=0; j < infoHeader.width; j++) myfile << (int)g[i][j] << " ";
             myfile << "\n";
        }
        myfile << "\n";
        myfile << "Blue data:\n";
        for(int i=0; i < infoHeader.height; i++){
            for(int j=0; j < infoHeader.width; j++) myfile << (int)b[i][j] << " ";
             myfile << "\n";
        }
        
        myfile.close();
    }
    else cout << "Unable to open file";
    return 0;
    


}


/*
   Read a possibly byte swapped unsigned short integer
*/
int ReadUShort(FILE *fptr,short unsigned *n,int swap)
{
   unsigned char *cptr,tmp;

   if (fread(n,2,1,fptr) != 1)
      return(0);
   if (swap) {
      cptr = (unsigned char *)n;
      tmp = cptr[0];
      cptr[0] = cptr[1];
      cptr[1] =tmp;
   }
   return(1);
}

/*
   Read a possibly byte swapped unsigned integer
*/
int ReadUInt(FILE *fptr,unsigned int *n,int swap)
{
   unsigned char *cptr,tmp;

   if (fread(n,4,1,fptr) != 1)
      return(0);
   if (swap) {
      cptr = (unsigned char *)n;
      tmp = cptr[0];
      cptr[0] = cptr[3];
      cptr[3] = tmp;
      tmp = cptr[1];
      cptr[1] = cptr[2];
      cptr[2] = tmp;
   }
   return(1);
}
