/////////////////////////////////////////////////////////////////////////////////
// This code belongs to "Energy Efficient Deep Learning” project group at      //
// Cornell University, Graduate School of Electrical and Computer Engineering. //
// This code is strictly for research and education. Do not use it for any     //
// commercial purposes.                                                        //
// Author: Jonathan Wu; Guangwei Chen                                          //
/////////////////////////////////////////////////////////////////////////////////
#ifndef BATCHNORM_RELU_H
#define BATCHNORM_RELU_H


int perform_BatchNorm (float* input_bn, const float y, const float* moving_variance, const float* gamma, 
						const float* moving_mean, const float* beta, int N, int O);

#endif 
