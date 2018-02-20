#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>


unsigned char* readBMP(char*);

unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    for(i = 0; i < size; i += 3)
    {
            unsigned char tmp = data[i];
            data[i] = data[i+2];
            data[i+2] = tmp;
    }
	//data should contain the (R, G, B) values of the pixels. The color of pixel (i, j) is stored at 
	//data[j * width + i], data[j * width + i + 1] and data[j * width + i + 2].
    return data;
}

int main(int argc, char **argv){

	unsigned char* raw_RGB = readBMP(argv[1]);
	
	//~ int i, j;
	for(int i=0; i<224; i++){
		for(int j=0; j<224; j++){
			
			printf("x: %d\t y: %d\t R: %d\t G: %d\t B: %d\n", i, j, raw_RGB[i + j*224], raw_RGB[i + j*224 + 1], raw_RGB[i + j*224 +2]);
		}
	}
	//const float test = 3.91774e-35;
	//const float test = 4;
	//printf("%e\n", test);

	return 0;
}
