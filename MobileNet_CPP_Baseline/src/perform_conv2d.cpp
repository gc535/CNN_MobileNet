/////////////////////////////////////////////////////////////////////////////////
// This code belongs to "Energy Efficient Deep Learning” project group at      //
// Cornell University, Graduate School of Electrical and Computer Engineering. //
// This code is strictly for research and education. Do not use it for any     //
// commercial purposes.                                                        //
// Author: Jonathan Wu; Guangwei Chen                                          //
/////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "perform_conv2d.h"
#include "main.h"



// void perform_conv2d (float input[][224][3], float output[][112][32], const float conv_weight_0[864], 
// 					 int O, int I, int N, int M, int K, int S){

void perform_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S){

	int kernel_size = K*K;
	int ifmap_size = I*I;
	int ofmap_size = O*O;

	int in_x;
	int in_y;

	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

	// perform convolution 
	for (int n = 0; n < N; n++) {//#iterate over number of output channels
		for (int m = 0; m < M; m++) {//#iterate over number of input channels
			for (int x = 0; x < O; x++) {//iterate over number of rows in output fmap
				for (int y = 0; y < O; y++) {//iterate over number of cols in output fmap
					for (int c = 0; c < K; c++) {//iterate through columns of kernel
						for (int r = 0; r < K; r++) {//iterate through rows of kernel
							//check stride of convolution
							if(S==1){
								in_x = x+c-1;
								in_y = y+r-1;
							}else{
								in_x = S*x+c;
								in_y = S*y+r;
							
							}
							//if out of bounds, do nothing
							if(in_x>=I || in_y>=I || in_x<0 || in_y<0){
								//no op
							}else{
								//maps multidimensional arrays to 1d array
								int i_index = in_x + (in_y) * I + m * ifmap_size;
								int o_index = x + y * O + n * ofmap_size;
								
								//gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
								int w_index = c + r * K + (n * M + m) * kernel_size;
								output[o_index] += input[i_index]*conv_weight[w_index];
								
								//output[x][y][n] += input[in_x][in_y][m] * conv_weight_0[w_index];
			
							}		
						}
					}
				}
			}
		}
	}
}

void perform_pointwise_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S){

	int kernel_size = K*K;
	int ifmap_size = I*I;
	int ofmap_size = O*O;

	int in_x;
	int in_y;

	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

	// perform convolution 
	for (int n = 0; n < N; n++) {//#iterate over number of output channels
		for (int m = 0; m < M; m++) {//#iterate over number of input channels
			for (int x = 0; x < O; x++) {//iterate over number of rows in output fmap
				for (int y = 0; y < O; y++) {//iterate over number of cols in output fmap
					for (int c = 0; c < K; c++) {//iterate through columns of kernel
						for (int r = 0; r < K; r++) {//iterate through rows of kernel
							//check stride of convolution			
							in_x = S*x+c;
							in_y = S*y+r;
							//if out of bounds, do nothing
							if(in_x>=I || in_y>=I || in_x<0 || in_y<0){
								//no op
							}else{
								//maps multidimensional arrays to 1d array
								int i_index = in_x + (in_y) * I + m * ifmap_size;
								int o_index = x + y * O + n * ofmap_size;
								
								//gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
								int w_index = c + r * K + (n * M + m) * kernel_size;
								output[o_index] += input[i_index]*conv_weight[w_index];
								
								//output[x][y][n] += input[in_x][in_y][m] * conv_weight_0[w_index];
			
							}		
						}
					}
				}
			}
		}
	}
}


void perform_depthwise_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S){

	int kernel_size = K*K;
	int ifmap_size = I*I;
	int ofmap_size = O*O;

	int in_x;
	int in_y;

	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

	// perform convolution 
	for (int n = 0; n < N; n++) {//#iterate over number of output channels
		for (int m = 0; m < M; m++) {//#iterate over number of input channels
			for (int x = 0; x < O; x++) {//iterate over number of rows in output fmap
				for (int y = 0; y < O; y++) {//iterate over number of cols in output fmap
					for (int c = 0; c < K; c++) {//iterate through columns of kernel
						for (int r = 0; r < K; r++) {//iterate through rows of kernel
							
							//check stride of convolution
							if(S==1){
								in_x = x+c-1;
								in_y = y+r-1;
							}else{
								in_x = S*x+c;
								in_y = S*y+r;
							}
							//if out out of bounds, do nothing
							if(in_x>=I || in_y>=I || in_x<0 || in_y<0){

							}else{
								int i_index = in_x + (in_y) * I + m * ifmap_size;
								int o_index = x + y * O + m * ofmap_size;
								//fprintf(stderr, "o_index: %d\n", o_index);
								
								//gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
								int w_index = c + r * K + (n * M + m) * kernel_size;
								output[o_index] += input[i_index]*conv_weight[w_index];
								//fprintf(stderr, "in x: %d \t in y: %d\n", in_x, in_y); 
								//fprintf(stderr, "out x: %d \t out y: %d\n", x, y); 
								//fprintf(stderr, "output: %f\n", output[o_index]);
								//output[x][y][n] += input[in_x][in_y][m] * conv_weight_0[w_index];
			
							}		
						}
					}
				}
			}
		}
	}
}


