#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>

#include "conv_layers.h"
//#include "mobilenets.h"
#include "mobilenets_updated.h"

/////////////////////////// FPGA SIDE ///////////////////////////////
/////////////////////////////////////////////////////////////////////
void conv3x3(   
    float* input, float* output, const float* conv_weight,
    int x, int y, int n, int m, int O, int I, int M, int S)
{
    int ifmap_size = I*I;
    int ofmap_size = O*O;

    int in_x;
    int in_y;
    int out_channel_idx = m;
    if(I == 224) out_channel_idx = n;    //this is first layer normal convolution

    for (int c = 0; c < KERNEL_3; c++) {//iterate through columns of kernel
        for (int r = 0; r < KERNEL_3; r++) {//iterate through rows of kernel
            //check stride of convolution
            if(S == 1){
                in_x = x+c-1;
                in_y = y+r-1;
            }else{
                in_x = S*x+c;
                in_y = S*y+r;
            }
            //only 
            if(in_x<I && in_y<I && in_x>=0 && in_y>=0){
                //maps multidimensional arrays to 1d array
                int i_index = in_x + (in_y) * I + m * ifmap_size;
                int o_index = x + y * O + out_channel_idx * ofmap_size;
                
                //gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
                int w_index = c + r * KERNEL_3 + (n * M + m) * KERNEL3X3_SIZE;
                output[o_index] += input[i_index]*conv_weight[w_index];
            }       
        }
    }
}


void tiling_conv (float* input, float* output, const float* conv_weight, //all matrix size needs recalculation
                    int row, int col, int n, int m, int O, int I, int M, int S)
{
    for(int x = 0; x < TIL_LEN; x++){
        for(int y = 0; y < TIL_LEN; y++){
            conv3x3(input, output, conv_weight, row+x, col+y, n, m, O, I, M, S);
        }
    }

}


void pointwise_tiling(float* input, float* output, const float* conv_weight,
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



//////// hardware tiling test on 1st layer ////////////////

void conv3x3_accel(   
    float input[15*15*32], float output[7*7*32], const float conv_weight[9216],
    int x, int y, int too, int tii, int n, int m, int O, int I, int M, int S, int DW)
{
    int ifmap_size = I*I;
    int ofmap_size = O*O;
    
    int o_index;

    int in_x;
    int in_y;

    //int out_channel_idx = m;
    //if(I == 224) out_channel_idx = n;    //this is first layer normal convolution

    for (int c = 0; c < KERNEL_3; c++) {//iterate through columns of kernel
        for (int r = 0; r < KERNEL_3; r++) {//iterate through rows of kernel

            in_x = S*x+c;
            in_y = S*y+r;

            //maps multidimensional arrays to 1d array
            int i_index = in_x + (in_y) * I + (tii-m) * ifmap_size;

            //if DW convolution, map output fmaps to same pixels as input fmaps
            if(DW) o_index = x + y * O + (tii-m) * ofmap_size;
            //if regular convolution map output fmaps in same fashion
            else o_index = x + y * O + (too-n) * ofmap_size;
            
            //gets mapping from 4D tensor to 1D array, 3 is kernel width, 9 is kernel size
            int w_index = c + r * KERNEL_3 + (too * M + tii) * KERNEL3X3_SIZE;
            //int w_index = c + r * K + (n * M + m) * kernel_size;
            output[o_index] += input[i_index]*conv_weight[w_index];       
        }
    }
}

void tiling_conv_accel (float input[15*15*32], float output[7*7*32], const float conv_weight[9216], //all matrix size needs recalculation
                    int row, int col, int n, int m, int I, int N, int M, int S, int DW)
{   
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


/////////////////////////// HOST SIDE ///////////////////////////////
/////////////////////////////////////////////////////////////////////

void perform_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S){

	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

    for(int row = 0; row < I; row+=TIL_LEN){
        for(int col = 0; col < I; col+=TIL_LEN){
        	for (int n = 0; n < N; n++) {//#iterate over number of output channels
        		for (int m = 0; m < M; m++) {//#iterate over number of input channels
        			tiling_conv(input, output, conv_weight, row, col, n, m, O, I, M, S);
        		}

	        }
        }
    }
}

void perform_pointwise_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S)
{
	int ifmap_size = I*I;
	int ofmap_size = O*O;
	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

    for(int row = 0; row < I; row+=TIL_LEN){
        for(int col = 0; col < I; col+=TIL_LEN){
        	// perform convolution 
        	for (int n = 0; n < N; n++) {//#iterate over number of output channels
        		for (int m = 0; m < M; m++) {//#iterate over number of input channels
        			//where data steam in
                    pointwise_tiling(input, output, conv_weight, row, col, n, m, O, I, M, ifmap_size, ofmap_size);
        		}
            //where data steam out
        	}
        }
    }
}

void logit_pointwise_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
                     int N, int M)
{
    // initialize output fmaps, 
    for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

    // perform convolution 
    for (int n = 0; n < N; n++) {//#iterate over number of output channels
        for (int m = 0; m < M; m++) {//#iterate over number of input channels
            //where data steam in
                    int w_index = (n * M + m);
                    output[n] += input[m]*conv_weight[w_index];
                } 
    //where data steam out
    }
}

void perform_depthwise_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S){

	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;


    for(int row = 0; row < I; row+=TIL_LEN){
        for(int col = 0; col < I; col+=TIL_LEN){
        	for (int n = 0; n < N; n++) {//#iterate over number of output channels
        		for (int m = 0; m < M; m++) {//#iterate over number of input channels
        			tiling_conv(input, output, conv_weight, row, col, n, m, O, I, M, S);
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

void perform_conv2d_hw (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
                     int O, int I, int N, int M, int K, int S){

    
    printf("HW Conv\n");
    
    // initialize output fmaps, 
    for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

    int ifmap_size = I*I;
    int ofmap_size = O*O;

    int up_N;
    int up_M;


    int load_in_len;
    float in_buff[15*15*TM];
    float out_buff[7*7*TN];

    int DW = 1;

    if(I==224) DW=0;//regular conv


    for(int row = 0; row < O; row+=TIL_LEN){//iterate over output fmaps
        for(int col = 0; col < O; col+=TIL_LEN){
            for (int n = 0; n < N; n+=TN) {//#iterate over number of output channels

                //get upper bounds for output fmaps, DW always has output fmap N=1
                if(n+TN>=N) up_N = N;
                else up_N = n + TN;

                for (int m = 0; m < M; m+=TM) {//#iterate over number of input channels

                    //get upper bounds for input fmaps
                    if(m+TM>=M) up_M = M;
                    else up_M = m + TM;

                    //if stride 2, load in windows 15x15, otherwise 9x9 windows
                    if(S==2) load_in_len = 15;
                    else load_in_len = 9;
                    
                    load_input_buffer(in_buff, out_buff, input, load_in_len, row, col, ifmap_size, m, up_M, I, S);

                        
                    //for(int i=0; i<(7*7*32); i++) out_buff[i] = 0; //initialize out buffer
                    tiling_conv_accel(in_buff, out_buff, conv_weight, row, col, n, m, load_in_len, N, M, S, DW);
                    
                    //write output tile to global output fmap
                    //void write_output_buffer(float* output_buff, float* global_output, int row, int col, int m, int n, int DW, int ofmap_size, up_M, up_N){
                    write_output_buffer(out_buff, output, row, col, m, n, DW, ofmap_size, up_M, up_N, O);
   
   
                }//end input m
            }//end output n 
        }//end col 
    }//end row
}//end function


////////////////////  convolution helper////////////////
/////////////// loading streaming buffer //////////////
void load_input_buffer(float* input_buff, float* output_buff, float* global_input, int load_in_len, int row, int col, int ifmap_size, int m, int up_M, int I, int S){
    
    int in_x, in_y;
    
    for(int i=0; i<load_in_len; i++){
        for(int j=0; j<load_in_len; j++){
            if(S==1){
                in_x = row+i-1;
                in_y = col+j-1;
            }else{
                in_x = S*row+i;
                in_y = S*col+j;
            }
            
            
            //load in input fmap to 
            for(int load_in=m; load_in<up_M; load_in++){
                //global index for input fmaps
                int i_index = in_x + (in_y * I) + load_in * ifmap_size;
                
                //local index for input fmaps
                int local_in_index = i + j*load_in_len + (load_in-m) * (load_in_len*load_in_len);
                
                //pad the buffer if loading in global values that are out of bounds
                if(in_x>=I || in_y>=I || in_x<0 || in_y<0) input_buff[local_in_index] = 0;
                else input_buff[local_in_index] = global_input[i_index];            

            }

        }
    }
    //initialize out buffer
    for(int i=0; i<(7*7*TN); i++) output_buff[i] = 0; //initialize out buffer
}


void write_output_buffer(float* output_buff, float* global_output, int row, int col, int m, int n, int DW, int ofmap_size, int up_M, int up_N, int O){
    int local_out_index, load_out;
    //if depthwise conv, use input fmap indices to load output fmaps b/c it's only 1 channel @ a time
    if(DW){
        load_out = m;
        up_N = up_M;
    }
    else{
        load_out = n;
    }
    
    //stream out output fmaps
    for(int i=0; i<TIL_LEN; i++){
        for(int j=0; j<TIL_LEN; j++){
            for(; load_out<up_N; load_out++){
                //printf("i: %d\t j: %d\t load_out: %d\n", i, j, load_out);
                //printf("X: %d\t Y: %d\n", (row+i), (col+j) );
                int o_index = (row+i) + (col+j) * O + load_out * ofmap_size;

                //different mappings for DW and regular convolution
                if(DW) local_out_index = i + j * TIL_LEN + (load_out-m) * TIL_SIZE;
                else local_out_index = i + j * TIL_LEN + (load_out-n) * TIL_SIZE;
                
                global_output[o_index] = output_buff[local_out_index];
                //printf("%f\n", output[o_index]);
            }

            //reset load_out     
            if(DW){
                load_out = m;
                up_N = up_M;
            }else{
                load_out = n;
            }   
                                 
        }
    }   
}