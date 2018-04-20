#ifndef MOBILENETS_H
#define MOBILENETS_H

//Layer 0 Weights
#define conv_weight_0 864 

#define moving_variance_0 32  

#define moving_mean_0 32 

#define beta_0 32  

#define gamma_0 32  


//Layer 1 Depthwise Weights
#define depth_conv_weight_1 288  

#define moving_variance_depth_1 32  
#define moving_mean_depth_1 32  

#define beta_depth_1 32  

#define gamma_depth_1 32  


//Layer 1 Pointwise Weights
#define point_conv_weight_1 2048  

#define moving_variance_point_1 64 

#define moving_mean_point_1 64  

#define beta_point_1 64  

#define gamma_point_1 64  

//LAYER 2 Depthwise Weights
#define depth_conv_weight_2 576  

#define moving_variance_depth_2 64 

#define moving_mean_depth_2 64 

#define beta_depth_2 64  

#define gamma_depth_2 64  

//Layer 2 Pointwise Weights
#define point_conv_weight_2 8192  
#define moving_variance_point_2 128  

#define moving_mean_point_2 128  

#define beta_point_2 128  

#define gamma_point_2 128  

//LAYER 3 Depthwise Weights
#define depth_conv_weight_3 1152  

#define moving_variance_depth_3 128  

#define moving_mean_depth_3 128  

#define beta_depth_3 128  

#define gamma_depth_3 128  

//Layer 3 Pointwise Weights
#define point_conv_weight_3 16384  

#define moving_variance_point_3 128  

#define moving_mean_point_3 128  

#define beta_point_3 128  

#define gamma_point_3 128  

//LAYER 4 Depthwise Weights
#define depth_conv_weight_4 1152  
#define moving_variance_depth_4 128  

#define moving_mean_depth_4 128  

#define beta_depth_4 128  

#define gamma_depth_4 128  

//Layer 4 Pointwise Weights
#define point_conv_weight_4 32768  

#define moving_variance_point_4 256  

#define moving_mean_point_4 256  

#define beta_point_4 256  

#define gamma_point_4 256  

//LAYER 5 Depthwise Weights
#define depth_conv_weight_5 2304  

#define moving_variance_depth_5 256  

#define moving_mean_depth_5 256  

#define beta_depth_5 256  

#define gamma_depth_5 256  

//Layer 5 Pointwise Weights
#define point_conv_weight_5 65536  

#define moving_variance_point_5 256 

#define moving_mean_point_5 256  

#define beta_point_5 256  

#define gamma_point_5 256  


//LAYER 6 Depthwise Weights
#define depth_conv_weight_6 2304 

#define moving_variance_depth_6 256  

#define moving_mean_depth_6 256  

#define beta_depth_6 256  

#define gamma_depth_6 256  

//Layer 6 Pointwise Weights
#define point_conv_weight_6 131072  

#define moving_variance_point_6 512  

#define moving_mean_point_6 512  

#define beta_point_6 512  

#define gamma_point_6 512  

//LAYER 7 Depthwise Weights
#define depth_conv_weight_7 4608  

#define moving_variance_depth_7 512  

#define moving_mean_depth_7 512  

#define beta_depth_7 512  

#define gamma_depth_7 512  

//Layer 7 Pointwise Weights
#define point_conv_weight_7 262144  

#define moving_variance_point_7 512  

#define moving_mean_point_7 512  

#define beta_point_7 512  

#define gamma_point_7 512

//LAYER 8 Depthwise Weights
#define depth_conv_weight_8 4608

#define moving_variance_depth_8 512

#define moving_mean_depth_8 512

#define beta_depth_8 512

#define gamma_depth_8 512

//Layer 8 Pointwise Weights
#define point_conv_weight_8 262144

#define moving_variance_point_8 512

#define moving_mean_point_8 512

#define beta_point_8 512

#define gamma_point_8 512

//LAYER 9 Depthwise Weights
#define depth_conv_weight_9 4608

#define moving_variance_depth_9 512

#define moving_mean_depth_9 512

#define beta_depth_9 512

#define gamma_depth_9 512

//Layer 9 Pointwise Weights
#define point_conv_weight_9 262144

#define moving_variance_point_9 512

#define moving_mean_point_9 512

#define beta_point_9 512

#define gamma_point_9 512

//LAYER 10 Depthwise Weights
#define depth_conv_weight_10 4608

#define moving_variance_depth_10 512

#define moving_mean_depth_10 512

#define beta_depth_10 512

#define gamma_depth_10 512

//Layer 10 Pointwise Weights
#define point_conv_weight_10 262144

#define moving_variance_point_10 512

#define moving_mean_point_10 512

#define beta_point_10 512

#define gamma_point_10 512

//LAYER 11 Depthwise Weights
#define depth_conv_weight_11 4608

#define moving_variance_depth_11 512

#define moving_mean_depth_11 512

#define beta_depth_11 512

#define gamma_depth_11 512

//Layer 11 Pointwise Weights
#define point_conv_weight_11 262144

#define moving_variance_point_11 512

#define moving_mean_point_11 512

#define beta_point_11 512

#define gamma_point_11 512

//LAYER 12 Depthwise Weights
#define depth_conv_weight_12 4608

#define moving_variance_depth_12 512

#define moving_mean_depth_12 512

#define beta_depth_12 512

#define gamma_depth_12 512

//Layer 12 Pointwise Weights
#define point_conv_weight_12 524288

#define moving_variance_point_12 1024

#define moving_mean_point_12 1024

#define beta_point_12 1024

#define gamma_point_12 1024

//LAYER 13 Depthwise Weights
#define depth_conv_weight_13 9216

#define moving_variance_depth_13 1024

#define moving_mean_depth_13 1024

#define beta_depth_13 1024

#define gamma_depth_13 1024

//Layer 13 Pointwise Weights
#define point_conv_weight_13 1048576

#define moving_variance_point_13 1024

#define moving_mean_point_13 1024

#define beta_point_13 1024

#define gamma_point_13 1024

//Logit Layer
#define logit_conv_weight 1025024

#define logit_bias_weight 1001

//const int mean = 0;
//const int st_dev= 255;

//constants for max 1D array sizes
const int MAX_FMAP_SIZE = 802816; //max feature map size
const int MAX_W_CONV = 1024*1024; //max convolution filter 
const int MAX_GAMMA = 1024;
const int MAX_MOVING_MEAN = 1024;
const int MAX_BETA = 1024;
const int MAX_MOVING_VARIANCE = 1024;



int main(int argc,char **argv);
void mobilenets(float* mem_conv1, float* mem_conv2);




#endif
