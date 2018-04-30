/////////////////////////////////////////////////////////////////////////////////
// This code belongs to "Energy Efficient Deep Learning” project group at      //
// Cornell University, Graduate School of Electrical and Computer Engineering. //
// This code is strictly for research and education. Do not use it for any     //
// commercial purposes.                                                        //
// Author: Jonathan Wu; Guangwei Chen                                          //
/////////////////////////////////////////////////////////////////////////////////
#ifndef PERFORM_CONV_H
#define PERFORM_CONV_H



// void perform_conv2d (float input[][224][3], float output[][112][32], const float conv_weight[864], 
// 					 int O, int I, int N, int M, int K, int S);

void perform_conv2d (float* input, float* output, const float* conv_weight, 
					 int O, int I, int N, int M, int K, int S);
					 
void perform_pointwise_conv2d (float* input, float* output, const float* conv_weight, 
					 int O, int I, int N, int M, int K, int S);
					 
void perform_depthwise_conv2d (float* input, float* output, const float* conv_weight, 
					 int O, int I, int N, int M, int K, int S);




#endif
