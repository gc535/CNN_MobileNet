#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>



void perform_Depthwise ( float input[][112][32] ){
    #include "data.h"
    float output[112][112][32];
	//BEGINNING OF 2D Depthwise Conv layer
	// initialize output fmaps
	for (int i = 0; i < 112; i++){
		for(int j=0; j < 112; j++){
			for(int k=0; k<32; k++){
				output[i][j][k]=0;
			}
		}
	}

	//perform depthwise convolution
	for (int n = 0; n < 1; n++) {//#channel multiplier
		for (int m = 0; m < 32; m++) {//# input channels
			for (int x = 0; x < 112; x++) {//output fmap width
				for (int y = 0; y < 112; y++) {//output fmap height

					for (int c = 0; c < 3; c++) {//kernel width
						for (int r = 0; r < 3; r++) {//kernel height

							int input_x = x+c-1; //input for convolution of stride 1
							int input_y = y+r-1;

							if(input_x>=112 || input_y>=112 || input_x<0 || input_y<0){

							}else{
								//maping from 4D tensor to 1D array (3 is kernel width, 32 is #input channels, 9
								//is kernel size
								int w_index = c + r * 3 + (n * 32 + m) * 9;
								output[x][y][m] += input[input_x][input_y][m] * depth_conv_weight_1[w_index];
							}		
						}
					}
				}
			}
		}
	}



	for(int i=0; i<32; i++){
		std::cout << output[111][111][i] << "\n";
	}

}