#ifndef BATCHNORM_RELU_H
#define BATCHNORM_RELU_H


int perform_BatchNorm (float* input_bn, const float y, const float* moving_variance, const float* gamma, 
						const float* moving_mean, const float* beta, int N, int O);

#endif 