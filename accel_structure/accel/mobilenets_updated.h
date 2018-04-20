#ifndef MOBILENETS_H
#define MOBILENETS_H
//#include "Common.h"

//#define conv_weight_0 864


const char params_file[]  = "/params/lay_0_conv.zip";

//const int mean = 0;
//const int st_dev= 255;

//constants for max 1D array sizes
const int MAX_FMAP_SIZE = 802816; //max feature map size
const int MAX_W_CONV = 1024*1024; //max convolution filter 
const int MAX_GAMMA = 1024;
const int MAX_MOVING_MEAN = 1024;
const int MAX_BETA = 1024;
const int MAX_MOVING_VARIANCE = 1024;
const int TIL_SIZE = 49;
const int TIL_LEN = 7;
const int KERNEL3X3_SIZE = 9;
const int KERNEL_3 = 3;
const int TN = 64;
const int TM = 64;

//void load_weights(float*, int);
int main(int argc,char **argv);
void mobilenets(float* mem_conv1, float* mem_conv2);


//Layer 0 Weights
const float conv_weight_0[864] = {
	#include "mobilenets_weights/Conv2d_0_lay_0_conv_weights.dat"
};

const float moving_variance_0[32]  = {
	#include "mobilenets_weights/Conv2d_0_moving_variance.dat"
};

const float moving_mean_0[32] = {
	#include "mobilenets_weights/Conv2d_0_moving_mean.dat"	
};

const float beta_0[32] = {
	#include "mobilenets_weights/Conv2d_0_beta.dat"	
};

const float gamma_0[32] = {
	#include "mobilenets_weights/Conv2d_0_gamma.dat"
};


//Layer 1 Depthwise Weights
const float depth_conv_weight_1[288] = {
	#include "mobilenets_weights/Conv2d_1_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_1[32] = {
	#include "mobilenets_weights/Conv2d_1_depthwise_moving_variance.dat"
};

const float moving_mean_depth_1[32] = {
	#include "mobilenets_weights/Conv2d_1_depthwise_moving_mean.dat"
};

const float beta_depth_1[32] = {
	#include "mobilenets_weights/Conv2d_1_depthwise_beta.dat"
};

const float gamma_depth_1[32] = {
	#include "mobilenets_weights/Conv2d_1_depthwise_gamma.dat"
};


//Layer 1 Pointwise Weights
const float point_conv_weight_1[2048] = {
	#include "mobilenets_weights/Conv2d_1_pointwise_pointwise.dat"
};

const float moving_variance_point_1[64] = {
	#include "mobilenets_weights/Conv2d_1_pointwise_moving_variance.dat"
};

const float moving_mean_point_1[64] = {
	#include "mobilenets_weights/Conv2d_1_pointwise_moving_mean.dat"
};

const float beta_point_1[64] = {
	#include "mobilenets_weights/Conv2d_1_pointwise_beta.dat"
};

const float gamma_point_1[64] = {
	#include "mobilenets_weights/Conv2d_1_pointwise_gamma.dat"
};

//LAYER 2 Depthwise Weights
const float depth_conv_weight_2[576] = {
	#include "mobilenets_weights/Conv2d_2_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_2[64] = {
	#include "mobilenets_weights/Conv2d_2_depthwise_moving_variance.dat"
};

const float moving_mean_depth_2[64] = {
	#include "mobilenets_weights/Conv2d_2_depthwise_moving_mean.dat"
};

const float beta_depth_2[64] = {
	#include "mobilenets_weights/Conv2d_2_depthwise_beta.dat"
};

const float gamma_depth_2[64] = {
	#include "mobilenets_weights/Conv2d_2_depthwise_gamma.dat"
};

//Layer 2 Pointwise Weights
const float point_conv_weight_2[8192] = {
	#include "mobilenets_weights/Conv2d_2_pointwise_pointwise.dat"
};

const float moving_variance_point_2[128] = {
	#include "mobilenets_weights/Conv2d_2_pointwise_moving_variance.dat"
};

const float moving_mean_point_2[128] = {
	#include "mobilenets_weights/Conv2d_2_pointwise_moving_mean.dat"
};

const float beta_point_2[128] = {
	#include "mobilenets_weights/Conv2d_2_pointwise_beta.dat"
};

const float gamma_point_2[128] = {
	#include "mobilenets_weights/Conv2d_2_pointwise_gamma.dat"
};

//LAYER 3 Depthwise Weights
const float depth_conv_weight_3[1152] = {
	#include "mobilenets_weights/Conv2d_3_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_3[128] = {
	#include "mobilenets_weights/Conv2d_3_depthwise_moving_variance.dat"
};

const float moving_mean_depth_3[128] = {
	#include "mobilenets_weights/Conv2d_3_depthwise_moving_mean.dat"
};

const float beta_depth_3[128] = {
	#include "mobilenets_weights/Conv2d_3_depthwise_beta.dat"
};

const float gamma_depth_3[128] = {
	#include "mobilenets_weights/Conv2d_3_depthwise_gamma.dat"
};

//Layer 3 Pointwise Weights
const float point_conv_weight_3[16384] = {
	#include "mobilenets_weights/Conv2d_3_pointwise_pointwise.dat"
};

const float moving_variance_point_3[128] = {
	#include "mobilenets_weights/Conv2d_3_pointwise_moving_variance.dat"
};

const float moving_mean_point_3[128] = {
	#include "mobilenets_weights/Conv2d_3_pointwise_moving_mean.dat"
};

const float beta_point_3[128] = {
	#include "mobilenets_weights/Conv2d_3_pointwise_beta.dat"
};

const float gamma_point_3[128] = {
	#include "mobilenets_weights/Conv2d_3_pointwise_gamma.dat"
};

//LAYER 4 Depthwise Weights
const float depth_conv_weight_4[1152] = {
	#include "mobilenets_weights/Conv2d_4_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_4[128] = {
	#include "mobilenets_weights/Conv2d_4_depthwise_moving_variance.dat"
};

const float moving_mean_depth_4[128] = {
	#include "mobilenets_weights/Conv2d_4_depthwise_moving_mean.dat"
};

const float beta_depth_4[128] = {
	#include "mobilenets_weights/Conv2d_4_depthwise_beta.dat"
};

const float gamma_depth_4[128] = {
	#include "mobilenets_weights/Conv2d_4_depthwise_gamma.dat"
};

//Layer 4 Pointwise Weights
const float point_conv_weight_4[32768] = {
	#include "mobilenets_weights/Conv2d_4_pointwise_pointwise.dat"
};

const float moving_variance_point_4[256] = {
	#include "mobilenets_weights/Conv2d_4_pointwise_moving_variance.dat"
};

const float moving_mean_point_4[256] = {
	#include "mobilenets_weights/Conv2d_4_pointwise_moving_mean.dat"
};

const float beta_point_4[256] = {
	#include "mobilenets_weights/Conv2d_4_pointwise_beta.dat"
};

const float gamma_point_4[256] = {
	#include "mobilenets_weights/Conv2d_4_pointwise_gamma.dat"
};

//LAYER 5 Depthwise Weights
const float depth_conv_weight_5[2304] = {
	#include "mobilenets_weights/Conv2d_5_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_5[256] = {
	#include "mobilenets_weights/Conv2d_5_depthwise_moving_variance.dat"
};

const float moving_mean_depth_5[256] = {
	#include "mobilenets_weights/Conv2d_5_depthwise_moving_mean.dat"
};

const float beta_depth_5[256] = {
	#include "mobilenets_weights/Conv2d_5_depthwise_beta.dat"
};

const float gamma_depth_5[256] = {
	#include "mobilenets_weights/Conv2d_5_depthwise_gamma.dat"
};

//Layer 5 Pointwise Weights
const float point_conv_weight_5[65536] = {
	#include "mobilenets_weights/Conv2d_5_pointwise_pointwise.dat"
};

const float moving_variance_point_5[256] = {
	#include "mobilenets_weights/Conv2d_5_pointwise_moving_variance.dat"
};

const float moving_mean_point_5[256] = {
	#include "mobilenets_weights/Conv2d_5_pointwise_moving_mean.dat"
};

const float beta_point_5[256] = {
	#include "mobilenets_weights/Conv2d_5_pointwise_beta.dat"
};

const float gamma_point_5[256] = {
	#include "mobilenets_weights/Conv2d_5_pointwise_gamma.dat"
};


//LAYER 6 Depthwise Weights
const float depth_conv_weight_6[2304] = {
	#include "mobilenets_weights/Conv2d_6_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_6[256] = {
	#include "mobilenets_weights/Conv2d_6_depthwise_moving_variance.dat"
};

const float moving_mean_depth_6[256] = {
	#include "mobilenets_weights/Conv2d_6_depthwise_moving_mean.dat"
};

const float beta_depth_6[256] = {
	#include "mobilenets_weights/Conv2d_6_depthwise_beta.dat"
};

const float gamma_depth_6[256] = {
	#include "mobilenets_weights/Conv2d_6_depthwise_gamma.dat"
};

//Layer 6 Pointwise Weights
const float point_conv_weight_6[131072] = {
	#include "mobilenets_weights/Conv2d_6_pointwise_pointwise.dat"
};

const float moving_variance_point_6[512] = {
	#include "mobilenets_weights/Conv2d_6_pointwise_moving_variance.dat"
};

const float moving_mean_point_6[512] = {
	#include "mobilenets_weights/Conv2d_6_pointwise_moving_mean.dat"
};

const float beta_point_6[512] = {
	#include "mobilenets_weights/Conv2d_6_pointwise_beta.dat"
};

const float gamma_point_6[512] = {
	#include "mobilenets_weights/Conv2d_6_pointwise_gamma.dat"
};

//LAYER 7 Depthwise Weights
const float depth_conv_weight_7[4608] = {
	#include "mobilenets_weights/Conv2d_7_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_7[512] = {
	#include "mobilenets_weights/Conv2d_7_depthwise_moving_variance.dat"
};

const float moving_mean_depth_7[512] = {
	#include "mobilenets_weights/Conv2d_7_depthwise_moving_mean.dat"
};

const float beta_depth_7[512] = {
	#include "mobilenets_weights/Conv2d_7_depthwise_beta.dat"
};

const float gamma_depth_7[512] = {
	#include "mobilenets_weights/Conv2d_7_depthwise_gamma.dat"
};

//Layer 7 Pointwise Weights
const float point_conv_weight_7[262144] = {
	#include "mobilenets_weights/Conv2d_7_pointwise_pointwise.dat"
};

const float moving_variance_point_7[512] = {
	#include "mobilenets_weights/Conv2d_7_pointwise_moving_variance.dat"
};

const float moving_mean_point_7[512] = {
	#include "mobilenets_weights/Conv2d_7_pointwise_moving_mean.dat"
};

const float beta_point_7[512] = {
	#include "mobilenets_weights/Conv2d_7_pointwise_beta.dat"
};

const float gamma_point_7[512] = {
	#include "mobilenets_weights/Conv2d_7_pointwise_gamma.dat"
};

//LAYER 8 Depthwise Weights
const float depth_conv_weight_8[4608] = {
	#include "mobilenets_weights/Conv2d_8_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_8[512] = {
	#include "mobilenets_weights/Conv2d_8_depthwise_moving_variance.dat"
};

const float moving_mean_depth_8[512] = {
	#include "mobilenets_weights/Conv2d_8_depthwise_moving_mean.dat"
};

const float beta_depth_8[512] = {
	#include "mobilenets_weights/Conv2d_8_depthwise_beta.dat"
};

const float gamma_depth_8[512] = {
	#include "mobilenets_weights/Conv2d_8_depthwise_gamma.dat"
};

//Layer 8 Pointwise Weights
const float point_conv_weight_8[262144] = {
	#include "mobilenets_weights/Conv2d_8_pointwise_pointwise.dat"
};

const float moving_variance_point_8[512] = {
	#include "mobilenets_weights/Conv2d_8_pointwise_moving_variance.dat"
};

const float moving_mean_point_8[512] = {
	#include "mobilenets_weights/Conv2d_8_pointwise_moving_mean.dat"
};

const float beta_point_8[512] = {
	#include "mobilenets_weights/Conv2d_8_pointwise_beta.dat"
};

const float gamma_point_8[512] = {
	#include "mobilenets_weights/Conv2d_8_pointwise_gamma.dat"
};

//LAYER 9 Depthwise Weights
const float depth_conv_weight_9[4608] = {
	#include "mobilenets_weights/Conv2d_9_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_9[512] = {
	#include "mobilenets_weights/Conv2d_9_depthwise_moving_variance.dat"
};

const float moving_mean_depth_9[512] = {
	#include "mobilenets_weights/Conv2d_9_depthwise_moving_mean.dat"
};

const float beta_depth_9[512] = {
	#include "mobilenets_weights/Conv2d_9_depthwise_beta.dat"
};

const float gamma_depth_9[512] = {
	#include "mobilenets_weights/Conv2d_9_depthwise_gamma.dat"
};

//Layer 9 Pointwise Weights
const float point_conv_weight_9[262144] = {
	#include "mobilenets_weights/Conv2d_9_pointwise_pointwise.dat"
};

const float moving_variance_point_9[512] = {
	#include "mobilenets_weights/Conv2d_9_pointwise_moving_variance.dat"
};

const float moving_mean_point_9[512] = {
	#include "mobilenets_weights/Conv2d_9_pointwise_moving_mean.dat"
};

const float beta_point_9[512] = {
	#include "mobilenets_weights/Conv2d_9_pointwise_beta.dat"
};

const float gamma_point_9[512] = {
	#include "mobilenets_weights/Conv2d_9_pointwise_gamma.dat"
};

//LAYER 10 Depthwise Weights
const float depth_conv_weight_10[4608] = {
	#include "mobilenets_weights/Conv2d_10_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_10[512] = {
	#include "mobilenets_weights/Conv2d_10_depthwise_moving_variance.dat"
};

const float moving_mean_depth_10[512] = {
	#include "mobilenets_weights/Conv2d_10_depthwise_moving_mean.dat"
};

const float beta_depth_10[512] = {
	#include "mobilenets_weights/Conv2d_10_depthwise_beta.dat"
};

const float gamma_depth_10[512] = {
	#include "mobilenets_weights/Conv2d_10_depthwise_gamma.dat"
};

//Layer 10 Pointwise Weights
const float point_conv_weight_10[262144] = {
	#include "mobilenets_weights/Conv2d_10_pointwise_pointwise.dat"
};

const float moving_variance_point_10[512] = {
	#include "mobilenets_weights/Conv2d_10_pointwise_moving_variance.dat"
};

const float moving_mean_point_10[512] = {
	#include "mobilenets_weights/Conv2d_10_pointwise_moving_mean.dat"
};

const float beta_point_10[512] = {
	#include "mobilenets_weights/Conv2d_10_pointwise_beta.dat"
};

const float gamma_point_10[512] = {
	#include "mobilenets_weights/Conv2d_10_pointwise_gamma.dat"
};

//LAYER 11 Depthwise Weights
const float depth_conv_weight_11[4608] = {
	#include "mobilenets_weights/Conv2d_11_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_11[512] = {
	#include "mobilenets_weights/Conv2d_11_depthwise_moving_variance.dat"
};

const float moving_mean_depth_11[512] = {
	#include "mobilenets_weights/Conv2d_11_depthwise_moving_mean.dat"
};

const float beta_depth_11[512] = {
	#include "mobilenets_weights/Conv2d_11_depthwise_beta.dat"
};

const float gamma_depth_11[512] = {
	#include "mobilenets_weights/Conv2d_11_depthwise_gamma.dat"
};

//Layer 11 Pointwise Weights
const float point_conv_weight_11[262144] = {
	#include "mobilenets_weights/Conv2d_11_pointwise_pointwise.dat"
};

const float moving_variance_point_11[512] = {
	#include "mobilenets_weights/Conv2d_11_pointwise_moving_variance.dat"
};

const float moving_mean_point_11[512] = {
	#include "mobilenets_weights/Conv2d_11_pointwise_moving_mean.dat"
};

const float beta_point_11[512] = {
	#include "mobilenets_weights/Conv2d_11_pointwise_beta.dat"
};

const float gamma_point_11[512] = {
	#include "mobilenets_weights/Conv2d_11_pointwise_gamma.dat"
};

//LAYER 12 Depthwise Weights
const float depth_conv_weight_12[4608] = {
	#include "mobilenets_weights/Conv2d_12_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_12[512] = {
	#include "mobilenets_weights/Conv2d_12_depthwise_moving_variance.dat"
};

const float moving_mean_depth_12[512] = {
	#include "mobilenets_weights/Conv2d_12_depthwise_moving_mean.dat"
};

const float beta_depth_12[512] = {
	#include "mobilenets_weights/Conv2d_12_depthwise_beta.dat"
};

const float gamma_depth_12[512] = {
	#include "mobilenets_weights/Conv2d_12_depthwise_gamma.dat"
};

//Layer 12 Pointwise Weights
const float point_conv_weight_12[524288] = {
	#include "mobilenets_weights/Conv2d_12_pointwise_pointwise.dat"
};

const float moving_variance_point_12[1024] = {
	#include "mobilenets_weights/Conv2d_12_pointwise_moving_variance.dat"
};

const float moving_mean_point_12[1024] = {
	#include "mobilenets_weights/Conv2d_12_pointwise_moving_mean.dat"
};

const float beta_point_12[1024] = {
	#include "mobilenets_weights/Conv2d_12_pointwise_beta.dat"
};

const float gamma_point_12[1024] = {
	#include "mobilenets_weights/Conv2d_12_pointwise_gamma.dat"
};

//LAYER 13 Depthwise Weights
const float depth_conv_weight_13[9216] = {
	#include "mobilenets_weights/Conv2d_13_depthwise_depthwise_weights.dat"
};

const float moving_variance_depth_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_depthwise_moving_variance.dat"
};

const float moving_mean_depth_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_depthwise_moving_mean.dat"
};

const float beta_depth_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_depthwise_beta.dat"
};

const float gamma_depth_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_depthwise_gamma.dat"
};

//Layer 13 Pointwise Weights
const float point_conv_weight_13[1048576] = {
	#include "mobilenets_weights/Conv2d_13_pointwise_pointwise.dat"
};

const float moving_variance_point_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_pointwise_moving_variance.dat"
};

const float moving_mean_point_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_pointwise_moving_mean.dat"
};

const float beta_point_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_pointwise_beta.dat"
};

const float gamma_point_13[1024] = {
	#include "mobilenets_weights/Conv2d_13_pointwise_gamma.dat"
};

//Logit Layer
const float logit_conv_weight[1025024] = {
	#include "mobilenets_weights/logit_conv_weights.dat"
};

const float logit_bias_weight[1001] = {
	#include "mobilenets_weights/logit_bias.dat"
};

#endif
