#include <stdio.h>
#include <string.h>
#include <math.h>



void perform_conv2d ( float input[][224][3], float output[][112][32] ){
    #include "data.h"
	//CONVOLUTION 
	int O = 112; //output feature map width
	int I = 224; //input feature map width
	int N=32; //number of output feature maps
	int M=3; //number of input feature maps
    int K=3; //kernel width

	// initialize output fmaps
	for (int i = 0; i < 112; i++){
		for(int j=0; j < 112; j++){
			for(int k=0; k<32; k++){
				output[i][j][k]=0;
			}
		}
	}
	// perform convolution 
	for (int n = 0; n < N; n++) {//#output maps
		for (int m = 0; m < M; m++) {//# input channels
			for (int x = 0; x < O; x++) {
				for (int y = 0; y < O; y++) {

					for (int c = 0; c < K; c++) {
						for (int r = 0; r < K; r++) {

							int input_x = 2*x+c;
							int input_y = 2*y+r;

							if(input_x==224 || input_y==224){

							}else{
								//gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
								int w_index = c + r * 3 + (n * 3 + m) * 9;
								output[x][y][n] += input[input_x][input_y][m] * conv_weight_0[w_index];
			
							}		
						}
					}
				}
			}
		}
	}
}




