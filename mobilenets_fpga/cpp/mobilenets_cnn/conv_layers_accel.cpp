/*
(c) Copyright 2013 - 2016 Xilinx, Inc. All rights reserved. 

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER 
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising
under or in connection with these materials, including for any direct, or any
indirect, special, incidental, or consequential loss or damage (including loss
of data, profits, goodwill, or any type of loss or damage suffered as a result
of any action brought by a third party) even if such damage or loss was
reasonably foreseeable or Xilinx had been advised of the possibility of the
same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES. 
*/

#include <stdio.h>
#include <stdlib.h>

#include "conv_layers_accel.h"


/**
 *
 * Design principles to achieve II = 1
 * 1. Stream data into local RAM for inputs (multiple access required)
 * 2. Partition local RAMs into N/2 sub-arrays for fully parallel access (dual-port read)
 * 3. Pipeline the dot-product loop, to fully unroll it
 * 4. Separate multiply-accumulate in inner loop to force two FP operators
 *
 
void mmult_accel(float A[N*N], float B[N*N], float C[N*N]) 
{
     float A_tmp[N][N], B_tmp[N][N];
#pragma HLS array_partition variable=A_tmp block factor=16 dim=2
#pragma HLS array_partition variable=B_tmp block factor=16 dim=1
     
     for(int i=0; i<N; i++) {
          for(int j=0; j<N; j++) {
#pragma HLS PIPELINE
               A_tmp[i][j] = A[i * N + j];
               B_tmp[i][j] = B[i * N + j];
          }
     }
     
     for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE
               float result = 0;
               for (int k = 0; k < N; k++) {
                    float term = A_tmp[i][k] * B_tmp[k][j];
                    result += term;
               }
               C[i * N + j] = result;
          }
     }
}
*/


/////////////////////////// FPGA SIDE ///////////////////////////////
/////////////////////////////////////////////////////////////////////

//tiling_conv_accel(in_buff, out_buff, conv_weight, row, col, n, m, TIL_LEN, load_in_len, M, S, DW);

//Takes in 15x15x32 or 9x9x32 tiles and performs convolution on the tile to calculate partial output fmap
void tiling_conv_accel (float input[15*15*TM], float output[7*7*TN], const float conv_weight[9216], //all matrix size needs recalculation
                    int row, int col, int n, int m, int I, int N, int M, int S, int DW)
{
	//printf("Entering Tiling Function \n");
    static float weights[9216];
    float input_buff[15*15*32];
    float output_buff[7*7*32];
    
    
    int upper_M;
    int upper_N;

    if(m+TM >= M){
		//printf("setting upper_M");
		upper_M = M;
	}
    else upper_M = m+TM;

    if(n+TN >= N) upper_N = N;
    else upper_N = n+TN;



	//printf("DW: %d\n", DW);
	//printf("m: %d\t n: %d\t Upper N: %d \t Upper M: %d\n", M, n, upper_N, upper_M);
    
    for(int x = 0; x < TIL_LEN; x++){
        for(int y = 0; y < TIL_LEN; y++){
            for(int too=n; too < upper_N; too++){
                for(int tii= m; tii < upper_M; tii++){
					
					//printf("input channel: %d\t output channel %d\n", tii, too);
                    conv3x3_accel(input, output, conv_weight, x, y, too, tii, n, m, TIL_LEN, I, M, S, DW);
                    
				}
            }
        }
    }

}

void conv3x3_accel(   
    float input[15*15*TM], float output[7*7*TN], const float conv_weight[9216],
    int x, int y, int too, int tii, int n, int m, int O, int I, int M, int S, int DW)
{
	
	//printf("Entering Conv Function \n");
    int ifmap_size = I*I;
    int ofmap_size = O*O;
    
    int o_index;

    int in_x;
    int in_y;

    int out_channel_idx = m;
    if(I == 224) out_channel_idx = n;    //this is first layer normal convolution

    for (int c = 0; c < KERNEL_3; c++) {//iterate through columns of kernel
        for (int r = 0; r < KERNEL_3; r++) {//iterate through rows of kernel
            //check stride of convolution
            //if(S == 1){
            //    in_x = x+c-1;
            //    in_y = y+r-1;
            //}else{

            //padding the input buffer now so no longer need no ops
            in_x = S*x+c;
            in_y = S*y+r;
            //}
            //only 
            //if(in_x<I && in_y<I && in_x>=0 && in_y>=0){

            //maps multidimensional arrays to 1d array
            int i_index = in_x + (in_y) * I + (tii-m) * ifmap_size;

            //if DW convolution, map output fmaps to same pixels as input fmaps
            if(DW) o_index = x + y * O + (tii-m) * ofmap_size;
            //if regular convolution map output fmaps in same fashion
            else o_index = x + y * O + (too-n) * ofmap_size;
            
            //printf("input channel: %d\t output channel: %d\n", tii, too);
            
            //gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
            int w_index = c + r * KERNEL_3 + (too * M + tii) * KERNEL3X3_SIZE;
            //int w_index = c + r * K + (n * M + m) * kernel_size;
            output[o_index] += input[i_index]*conv_weight[w_index];
            //printf("x: %d\t y: %d\t output: %f\t tmp: %f\n", x, y, output[o_index], tmp);

            //}       
        }
    }
}

void batchnorm_accel(float input[7*7*TN], const float add_bias[1024], const float mult_bias[1024], int n, int up_N){

		for(int i=0; i<TIL_LEN; i++){
			for(int j=0; j<TIL_LEN; j++){
				for(int curr_channel=n; curr_channel<up_N; curr_channel++){
					
					//perform BN calculations
					int local_in_index = i + j*TIL_LEN + (curr_channel-n)*TIL_SIZE;
					input[local_in_index] = input[local_in_index]*mult_bias[curr_channel] + add_bias[curr_channel];
					
					//perform ReLU6 
					input[local_in_index] = (input[local_in_index] > 0) ? input[local_in_index] : 0;
					input[local_in_index] = (input[local_in_index] <= 6) ? input[local_in_index] : 6;
					
				}//end curr channel
			}//end j
		}//end i
	
}
/*
void pointwise_tiling_accel(float* input, float* output, const float* conv_weight,
                    int row, int col, int n, int m, int O, int I, int M, int ifmap_size, int ofmap_size)
{
    for (int x = 0; x < TIL_LEN; x++) {//iterate over number of rows in output fmap
        for (int y = 0; y < TIL_LEN; y++) {//iterate over number of cols in output fmap
            //maps multidimensional arrays to 1d array
            int i_index = x+row + (y+col) * I + m * ifmap_size;
            int o_index = x+row + (y+col) * O + n * ofmap_size;
            //gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
            int w_index = (n * M + m);
            output[o_index] += input[i_index]*conv_weight[w_index];
        }
    }
}
*/
