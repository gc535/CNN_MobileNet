#include <string.h>

#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include "ParamIO.h"
#include "Timer.h"
#include "Common.h"

//#include "sds_lib.h"

//#include "conv_layers.h"
#include "conv_layers_accel.h"
#include "loadbmp.h"
//#include "mobilenets_updated.h"

using namespace std;

//constants for max 1D array sizes
const int MAX_FMAP_SIZE = 802816; //max feature map size
const int MAX_W_CONV = 1024*1024; //max convolution filter 
const int MAX_GAMMA = 1024;
const int MAX_MOVING_MEAN = 1024;
const int MAX_BETA = 1024;
const int MAX_MOVING_VARIANCE = 1024;


const char params_conv0_file[]  = "/params/lay_0_conv.zip";
const char params_mult0_file[]  = "/params/lay_0_mult.zip";
const char params_add0_file[]  = "/params/lay_0_add.zip";


//cast int -> string type
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()
        


/*
class perf_counter
{
public:
     uint64_t tot, cnt, calls;
     perf_counter() : tot(0), cnt(0), calls(0) {};
     inline void reset() { tot = cnt = calls = 0; }
     inline void start() { cnt = sds_clock_counter(); calls++; };
     inline void stop() { tot += (sds_clock_counter() - cnt); };
     inline uint64_t avg_cpu_cycles() { return (tot / calls); };
};
*/

/////////////////////////// SW Implementation ///////////////////////////////
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


/////////////////////////// HOST SIDE ///////////////////////////////
/////////////////////////////////////////////////////////////////////

void perform_conv2d_sw (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
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

/*
void perform_conv2d_sw (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
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

*/

//HELPER FUNCTIONS FOR TILING ON HW

//load in input fmaps and clear output buff
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


//function call for regular convolution and depthwise convolution
//depthwise convolution rights to same channels as input 
void perform_conv2d_hw (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], 
					 int O, int I, int N, int M, int K, int S){

	
	printf("HW Conv\n");
	
	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;

    int ifmap_size = I*I;
    int ofmap_size = O*O;

    int in_x;
    int in_y;

    int up_N;
    int up_M;


    int load_in_len;
    int load_out;
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
                    
                    //printf("Current X: %d\t Current Y: %d\n", row, col);
					//int num_pixels_loaded = 0;
					
					//void load_input_buffer(float* input_buff, float* output_buff, float* global_input, int load_in_len, int row, int col, int ifmap_size, int m, int up_M)
					
					load_input_buffer(in_buff, out_buff, input, load_in_len, row, col, ifmap_size, m, up_M, I, S);
					/*
                    //load in input fmaps and clear output buff
                    for(int i=0; i<load_in_len; i++){
                        for(int j=0; j<load_in_len; j++){
                            if(S==1){
                                in_x = row+i-1;
                                in_y = col+j-1;
                            }else{
                                in_x = S*row+i;
                                in_y = S*col+j;
                            }
                            
                            printf("in_x: %d\t in_y: %d\n", in_x, in_y);
                            //load in input fmap to 
                            for(int load_in=m; load_in<up_M; load_in++){
                                //global index for input fmaps
                                printf("loading in input channel: %d\n", load_in); 
                                //printf("Global x: %d\t Global y: %d\t Input Channel: %d\n", in_x, in_y, load_in);
                                int i_index = in_x + (in_y * I) + load_in * ifmap_size;
                                //local index for input fmaps
                                int local_in_index = i + j*load_in_len + (load_in-m) * (load_in_len*load_in_len);
                                //pad the buffer if loading in global values that are out of bounds
                                if(in_x>=I || in_y>=I || in_x<0 || in_y<0) in_buff[local_in_index] = 0;
                                else in_buff[local_in_index] = input[i_index];
                                
                                

                            }

                        }
                    }
					*/
						
					//for(int i=0; i<(7*7*32); i++) out_buff[i] = 0; //initialize out buffer
        			tiling_conv_accel(in_buff, out_buff, conv_weight, row, col, n, m, load_in_len, N, M, S, DW);
        			
					//write output tile to global output fmap
					//void write_output_buffer(float* output_buff, float* global_output, int row, int col, int m, int n, int DW, int ofmap_size, up_M, up_N){
					write_output_buffer(out_buff, output, row, col, m, n, DW, ofmap_size, up_M, up_N, O);
   
/*
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
                                //int local_out_index = j + i * TIL_LEN + (load_out-n) * TIL_SIZE;
                                int local_out_index = i + j * TIL_LEN + (load_out-n) * TIL_SIZE;
                                output[o_index] = out_buff[local_out_index];
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
*/                    
                    
        		}//end input m
	        }//end output n 
        }//end col 
    }//end row
}//end function

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
void perform_BatchNorm_hw (float input[MAX_FMAP_SIZE], const float y, const float add_bias[MAX_MOVING_VARIANCE], const float mult_bias[MAX_GAMMA], int N, int O){

	float input_buff[7*7*TN];
	float output_buff[7*7*TN];
	
	int up_N;
	int load_n;

	int ofmap_size = O*O;


	
	for(int row=0; row<O; row+=TIL_LEN){
		for(int col=0; col<O; col+=TIL_LEN){
			for(int n=0; n<N; n+=TN){
				if(n+TN >= N) up_N = N;
				else up_N = n+TN;
				//Load input and send to FPGA
				for(int i=0; i<TIL_LEN; i++){
					for(int j=0; j<TIL_LEN; j++){
						for(load_n=n; load_n<up_N; load_n++){	
							//map global input to local input
							int i_index = (row+i) + (col+j)*O + load_n*(ofmap_size);
							int local_in_index = i + j*TIL_LEN + (load_n-n)*(TIL_SIZE);
							
							//load up input buffer
							input_buff[local_in_index] = input[i_index];
						}//end load_n
					}//end j
				}//end i
				
				//Call Batchnorm accel function
				batchnorm_accel(input_buff, add_bias, mult_bias, n, up_N);
				
				//load modified outputs back to global input
				for(int i=0; i<TIL_LEN; i++){
					for(int j=0; j<TIL_LEN; j++){
						for(load_n=n; load_n<up_N; load_n++){	
							//map global input to local input
							int i_index = (row+i) + (col+j)*O + load_n*(ofmap_size);
							int local_in_index = i + j*TIL_LEN + (load_n-n)*(TIL_SIZE);
							
							//load up input buffer
							input[i_index] = input_buff[local_in_index];
						}//end load_n
					}//end j
				}//end i				
					
			}//end channel
		}//end col
	}//end row
}


//This function recreates the batchnorm operations that tensorflow performs on the convolution outputs at each layer
void perform_BatchNorm_sw (float input[MAX_FMAP_SIZE], const float y, const float add_bias[MAX_MOVING_VARIANCE], const float mult_bias[MAX_GAMMA], int N, int O){


	int ofmap_size = O*O;


	float mul_output[N];

	for(int ele=0; ele<N; ele++){
		for(int i=0; i<O; i++){
			for(int j=0; j<O; j++){
				int i_index = i + j*O + ele*ofmap_size;
				//multiply and add biases
				input[i_index] = input[i_index]*mult_bias[ele] + add_bias[ele];
				//perform ReLU6
				input[i_index] = (input[i_index] > 0) ? input[i_index] : 0;
				input[i_index] = (input[i_index] <= 6) ? input[i_index] : 6;
				
			}
		}	
	}
}

static int result_check(float* output_sw, float* output_hw, int O, int N){
	int num_diff = 0;
	
	for(int i=0; i<O; i++){
		for(int j=0; j<O; j++){
			for(int n=0; n<N; n++){
				int o_ind = i + j*O + n*(O*O);
				if (output_sw[o_ind] != output_hw[o_ind]) {
				   std::cout << "Mismatch: data index=" << i << "d=" << output_sw[i] 
							 << ", dout=" << output_hw[i] << std::endl;
				   return 1;
				}		
			}
		}
	}
	//printf("num differences: %d\n", num_diff);	
	return 0;
}

int convolution_test(float *input_fmap,  float *conv_weights, float *output_sw, float *output_hw, float *add_bn_weights, float *mult_bn_weights)
{
	
	printf("Entering Conv Test\n");
    // std::cout << "Testing " << NUM_TESTS << " iterations of " << N << "x" << N 
    //           << " floating point mmult..." << std::endl;

     //perf_counter hw_ctr, sw_ctr;
     
     for (int i = 0; i < 1; i++) 
     {
          
          //perform_conv2d(mem_conv1, mem_conv2, conv_weight_0, 112, 224, 32, 3, 3, 2);
          
          //sw_ctr.start();
          perform_conv2d_sw(input_fmap, output_sw, conv_weights, 112, 224, 32, 3, 3, 2);
          //int perform_BatchNorm_sw (float input[MAX_FMAP_SIZE], const float y, const float add_bias[MAX_MOVING_VARIANCE], const float mult_bias[MAX_GAMMA], int N, int O)
          perform_BatchNorm_sw(output_sw, y_offset, add_bn_weights, mult_bn_weights, 32, 112);
          //sw_ctr.stop();

          //hw_ctr.start();
          perform_conv2d_hw(input_fmap, output_hw, conv_weights, 112, 224, 32, 3, 3, 2);
          //void perform_BatchNorm_hw (float input[MAX_FMAP_SIZE], const float y, const float add_bias[MAX_MOVING_VARIANCE], const float mult_bias[MAX_GAMMA], int N, int O)
          perform_BatchNorm_hw(output_hw, y_offset, add_bn_weights, mult_bn_weights, 32, 112);
          //hw_ctr.stop();
          
          /*
          printf("SW RESULTS\n");
          for(int i=0; i<32; i++){
			  int o_index = 107 + 111*112 + i*112*112;
			  printf("%f\n", output_sw[o_index]);
			  
		  }
          
          printf("\n");
          
          printf("HW RESULTS\n");
          for(int i=0; i<32; i++){
			  int o_index = 107 + 111*112 + i*112*112;
			  printf("%f\n", output_hw[o_index]);
			  
		  }
		  */
 
          if (result_check(output_sw, output_hw, 112, 32 ) )
               return 1;
     }
     /*
     uint64_t sw_cycles = sw_ctr.avg_cpu_cycles();
     uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
     double speedup = (double) sw_cycles / (double) hw_cycles;

     std::cout << "Average number of CPU cycles running mmult in software: "
               << sw_cycles << std::endl;
     std::cout << "Average number of CPU cycles running mmult in hardware: "
               << hw_cycles << std::endl;
     std::cout << "Speed up: " << speedup << std::endl;
	*/
     return 0;
}


//extern ostream cout;
void load_weights(const float* w, float* weights, int num_params){
	printf("Entering Load Weights\n");
	for(int i=0; i<num_params; i++){
		weights[i] = w[i];
	}
}



//main function
int main(int argc, char* argv[]){
     int test_passed = 0;

     
     float input_fmap[224*224*3];
     float output_hw[MAX_FMAP_SIZE];
     float output_sw[MAX_FMAP_SIZE];
     
     float conv_weights_0[3*3*3*32];
     float add_bias_0[32];
     float mult_bias_0[32];
     
    // const char params_conv0_file[]  = "/params/lay_0_conv.zip";
	//const char params_mult0_file[]  = "/params/lay_0_mult.zip";
	//const char params_add0_file[]  = "/params/lay_0_add.zip";

     
     
     //load weights
	 Params params0(get_root_dir() + params_conv0_file);
	 load_weights(params0.float_data(0), conv_weights_0, (3*3*3*32));
	 
	 Params params1(get_root_dir() + params_mult0_file);
	 load_weights(params1.float_data(0), mult_bias_0, 32);
	 
	 Params params2(get_root_dir() + params_add0_file);
	 load_weights(params2.float_data(0), add_bias_0, 32);
	 

	//load input fmap
	//declare 224x224 arrays to store input channels
	unsigned char b_input[224][224];
	unsigned char g_input[224][224];
	unsigned char r_input[224][224];
	
	string prefix;
    prefix = "/home/jzw8/resized_images/";
    string sufix;
    sufix = ".bmp";
    const char* filepath;   //file to open
    string target; 
    
    target = prefix + SSTR(0) + sufix;
    filepath=(target).c_str();
	//load BMP RGB array into input array
	load_bmp(filepath, b_input, g_input, r_input);
	
	for(int x = 0; x<224; x++){
		for(int y = 0; y < 224; y++){
			//1D array mapping(new implementation)
			int i_index_r = y + x*224 + 0*(224*224);
			int i_index_g = y + x*224 + 1*(224*224);
			int i_index_b = y + x*224 + 2*(224*224);
			input_fmap[i_index_r] = ((float)r_input[x][y]-0)/255;
			input_fmap[i_index_g] = ((float)g_input[x][y]-0)/255;
			input_fmap[i_index_b] = ((float)b_input[x][y]-0)/255;
		}
	}

   
     test_passed = convolution_test(input_fmap, conv_weights_0, output_sw, output_hw, add_bias_0, mult_bias_0);
     
     std::cout << "TEST " << (test_passed ? "FAILED" : "PASSED") << std::endl;
   
     return (test_passed ? -1 : 0);
}

