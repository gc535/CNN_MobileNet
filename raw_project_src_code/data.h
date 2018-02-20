#ifndef DATA_H
#define DATA_H

const float conv_weight_0[864] = {
	#include "test/Conv2d_0_lay_0_conv_weights.dat"
};

const float moving_variance_0[32]  = {
	#include "test/Conv2d_0_moving_variance.dat"
};

const float moving_mean_0[32] = {
	#include "test/Conv2d_0_moving_mean.dat"	
};

const float beta_0[32] = {
	#include "test/Conv2d_0_beta.dat"	
};

const float gamma_0[32] = {
	#include "test/Conv2d_0_gamma.dat"
};

const float depth_conv_weight_1[288] = {
	#include "test/Conv2d_1_depthwise_depthwise_weights.dat"
};


#endif