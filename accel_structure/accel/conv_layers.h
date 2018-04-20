#ifndef CONV_LAYERS_H
#define CONV_LAYERS_H


void conv3x3( float* input, float* output, const float* conv_weight,
    int x, int y, int n, int m, int O, int I, int M, int S);


// void perform_conv2d (float input[][224][3], float output[][112][32], const float conv_weight[864], 
// 					 int O, int I, int N, int M, int K, int S);
void tiling_conv(float* input, float* output, const float* conv_weight, 
                 int row, int col, int n, int m, int O, int I, int M, int S);

void pointwise_tiling(float* input, float* output, const float* conv_weight,
                    int row, int col, int n, int m, int O, int I, int M, int ifmap_size, int ofmap_size);

void perform_conv2d (float* input, float* output, const float* conv_weight, 
					 int O, int I, int N, int M, int K, int S);
					 
void perform_pointwise_conv2d (float* input, float* output, const float* conv_weight, 
					 int O, int I, int N, int M, int K, int S);
void logit_pointwise_conv2d (float* input, float* output, const float* conv_weight, 
                     int N, int M);
					 
void perform_depthwise_conv2d (float* input, float* output, const float* conv_weight, 
					 int O, int I, int N, int M, int K, int S);
					 
int perform_BatchNorm (float* input_bn, const float y, const float* moving_variance, const float* gamma, 
						const float* moving_mean, const float* beta, int N, int O);


////////hardware test//////

void conv3x3_accel(   
    float input[15*15*32], float output[7*7*32], const float conv_weight[9216],
    int x, int y, int too, int tii, int n, int m, int O, int I, int M, int S, int DW);


void perform_conv2d_hw (float* input, float* output, const float* conv_weight, 
                        int O, int I, int N, int M, int K, int S);


///////// conv_helper for streaming buffer loading ///////////
void load_input_buffer(float* input_buff, float* output_buff, float* mem_conv, int load_in_len, int row, int col, int ifmap_size, int m, int up_M, int I, int S);
void write_output_buffer(float* output_buff, float* global_output, int row, int col, int m, int n, int DW, int ofmap_size, int up_M, int up_N, int O);


#endif
