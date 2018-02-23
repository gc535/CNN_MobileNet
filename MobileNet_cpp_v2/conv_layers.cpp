#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


#include "conv_layers.h"
#include "mobilenets.h"



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

								
								//gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
								int w_index = c + r * K + (n * M + m) * kernel_size;
								output[o_index] += input[i_index]*conv_weight[w_index];
							}		
						}
					}
				}
			}
		}
	}
}

//This function recreates the batchnorm operations that tensorflow performs on the convolution outputs at each layer
int perform_BatchNorm (float input[MAX_FMAP_SIZE], const float y, const float moving_variance[MAX_MOVING_VARIANCE], const float gamma[MAX_GAMMA], 
						const float moving_mean[MAX_MOVING_MEAN], const float beta[MAX_BETA], int N, int O){

	float post_add_mv[N];
	float rec_sqrt[N];
	int ofmap_size = O*O;

	for(int ele=0; ele<N; ele++){
		post_add_mv[ele] = moving_variance[ele] + y;
		rec_sqrt[ele] = 1/sqrt(post_add_mv[ele]);
	}

	float mul_output[N];


	for(int ele=0; ele<N; ele++){
		mul_output[ele] = rec_sqrt[ele] * gamma[ele];
		for(int i=0; i<O; i++){
			for(int j=0; j<O; j++){

				//1D implementation
				int i_index = i + j * O + ele * ofmap_size;
				input[i_index] = input[i_index] * mul_output[ele];

			}
		} 
	}

	
	float mul_output_2[N];

	for(int ele=0; ele<N; ele++){
		mul_output_2[ele] = mul_output[ele] * moving_mean[ele];

	}


	float sub[N];

	for(int ele=0; ele<N; ele++){;
		sub[ele] = beta[ele] - mul_output_2[ele];
		for(int i=0; i<O; i++){
			for(int j=0; j<O; j++){

				//1D implementation
				int i_index = i + j * O + ele * ofmap_size;
				input[i_index] = input[i_index] + sub[ele];

			}
		}
	}


	//RELU6: if a value is < 0 it will be floored at 0 and if a value is >6 it will cut off at 6.0
	for(int i=0; i<O; i++){
		for(int j=0; j<O; j++){
			for(int k=0; k<N; k++){
				int i_index = i + j * O + k * ofmap_size;
				input[i_index] = (input[i_index] > 0) ? input[i_index] : 0;
				input[i_index] = (input[i_index] <= 6) ? input[i_index] : 6;
	
			}
		}
	}

	return 0;
}



