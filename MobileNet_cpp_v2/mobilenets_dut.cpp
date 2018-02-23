#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>


#include "mobilenets_dut.h"
#include "loadbmp.h"
#include "conv_layers.h"
#include "logit_layer.h"


//cast int -> string type
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()


class Layer_Helper{
    public:
        void get_intermediate(float input[MAX_FMAP_SIZE], int O, int N){
            int offset = (O-1) + (O-1) * O;  //last element of every output fmap, 
            for(int n=0; n<N; n++){
                int o_index =  offset + n*(112*112);     //indexing into last element of next fmap;
                fprintf(stderr, "%e\n", input[o_index]);
            }
    
        }
}; 



//TOP Function
void dut(
    hls::stream<bit32_t> &strm_in,
    hls::stream<bit32_t> &strm_out
)
{
  char *filePath;
  int prediction;

  //read in chars for filepath
  bit32_t input_lo = strm_in.read();
  bit32_t input_hi = strm_in.read();
  
  
  //load chars into filepath
  digit(31, 0) = input_lo(31,0);
  digit(48,32) = input_hi(16,0);

  // call mobilenets
  mobilenets(filePath, 0, 255, &prediction);

  // write out the result
  strm_out.write(prediction);
}




//function takes in 3 arguments, image path, mean and st deviation
void mobilenets(char* fp, int mean, int st_dev, int *prediction){
	
	//int mean = atoi(argv[2]);
	//int st_dev = atoi(argv[3]);
  	
	//declare 224x224 arrays to store input channels
	unsigned char b_input[224][224];
	unsigned char g_input[224][224];
	unsigned char r_input[224][224];

	//declare 2 1D arrays large enough to fix max FMAP size
    float mem_conv1[MAX_FMAP_SIZE];
    float mem_conv2[MAX_FMAP_SIZE];

    //load BMP RGB array into input array
    load_bmp(argc, argv, b_input, g_input, r_input);
        
    for(int x = 0; x<224; x++){
        for(int y = 0; y < 224; y++){

        	//1D array mapping(new implementation)
        	int i_index_r = y + x*224 + 0*(224*224);
        	int i_index_g = y + x*224 + 1*(224*224);
        	int i_index_b = y + x*224 + 2*(224*224);
        	mem_conv1[i_index_r] = ((float)r_input[x][y]-mean)/st_dev;
        	mem_conv1[i_index_g] = ((float)g_input[x][y]-mean)/st_dev;
        	mem_conv1[i_index_b] = ((float)b_input[x][y]-mean)/st_dev;
        }
    }
    
	


  	//layer 0
	perform_conv2d(mem_conv1, mem_conv2, conv_weight_0, 112, 224, 32, 3, 3, 2);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_0, gamma_0, moving_mean_0, beta_0, 32, 112);
    

	//layer 1 depthwise
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_1, 112, 112, 1, 32, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_1, gamma_depth_1, moving_mean_depth_1, beta_depth_1, 32, 112);
 	
 	
	//layer 1 pointwise
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_1, 112, 112, 64, 32, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_1, gamma_point_1, moving_mean_point_1, beta_point_1, 64, 112);

	//layer2 depthwise
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_2, 56, 112, 1, 64, 3, 2);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_2, gamma_depth_2, moving_mean_depth_2, beta_depth_2, 64, 56);

	//layer2 pointwise
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_2, 56, 56, 128, 64, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_2, gamma_point_2, moving_mean_point_2, beta_point_2, 128, 56);
	
	//layer3 depthwise
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_3, 56, 56, 1, 128, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_3, gamma_depth_3, moving_mean_depth_3, beta_depth_3, 128, 56);
	
	//layer3 pointwise
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_3, 56, 56, 128, 128, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_3, gamma_point_3, moving_mean_point_3, beta_point_3, 128, 56);
		
	//layer4 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_4, 28, 56, 1, 128, 3, 2);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_4, gamma_depth_4, moving_mean_depth_4, beta_depth_4, 128, 28);


	//layer4 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_4, 28, 28, 256, 128, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_4, gamma_point_4, moving_mean_point_4, beta_point_4, 256, 28);	
	

	//layer5 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_5, 28, 28, 1, 256, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_5, gamma_depth_5, moving_mean_depth_5, beta_depth_5, 256, 28);	
	

	//layer5 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_5, 28, 28, 256, 256, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_5, gamma_point_5, moving_mean_point_5, beta_point_5, 256, 28);	
		
	//layer6 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_6, 14, 28, 1, 256, 3, 2);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_6, gamma_depth_6, moving_mean_depth_6, beta_depth_6, 256, 14);	


	//layer6 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_6, 14, 14, 512, 256, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_6, gamma_point_6, moving_mean_point_6, beta_point_6, 512, 14);	

		
	//layer7 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_7, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_7, gamma_depth_7, moving_mean_depth_7, beta_depth_7, 512, 14);
		
	//layer7 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_7, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_7, gamma_point_7, moving_mean_point_7, beta_point_7, 512, 14);	
			
	//layer8 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_8, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_8, gamma_depth_8, moving_mean_depth_8, beta_depth_8, 512, 14);
		
	//layer 8 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_8, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_8, gamma_point_8, moving_mean_point_8, beta_point_8, 512, 14);
			
	//layer9 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_9, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_9, gamma_depth_9, moving_mean_depth_9, beta_depth_9, 512, 14);
		
	//layer9 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_9, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_9, gamma_point_9, moving_mean_point_9, beta_point_9, 512, 14);

	//layer10 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_10, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_10, gamma_depth_10, moving_mean_depth_10, beta_depth_10, 512, 14);
		
	//layer10 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_10, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_10, gamma_point_10, moving_mean_point_10, beta_point_10, 512, 14);
	

	//layer11 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_11, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_11, gamma_depth_11, moving_mean_depth_11, beta_depth_11, 512, 14);
		
	//layer11 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_11, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_11, gamma_point_11, moving_mean_point_11, beta_point_11, 512, 14);
			
	//layer12 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_12, 7, 14, 1, 512, 3, 2);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_12, gamma_depth_12, moving_mean_depth_12, beta_depth_12, 512, 7);
	
		
	//layer12 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_12, 7, 7, 1024, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_12, gamma_point_12, moving_mean_point_12, beta_point_12, 1024, 7);
	
			
	//layer13 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_13, 7, 7, 1, 1024, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_13, gamma_depth_13, moving_mean_depth_13, beta_depth_13, 1024, 7);
		
	//layer13 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_13, 7, 7, 1024, 1024, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_13, gamma_point_13, moving_mean_point_13, beta_point_13, 1024, 7);
	

  
    //LOGIT LAYER			 
    avg_pool2d(mem_conv2, mem_conv1, 1, 7, 1024, 1024, 7, 2);
    
    perform_pointwise_conv2d(mem_conv1, mem_conv2, logit_conv_weight, 1, 1, 1001, 1024, 1, 1);
 
    //Add Biases
    for(int i=0; i<1001; i++){
		mem_conv2[i] += logit_bias_weight[i];
	}

 
    //Predictions
    float softmax_denom = 0;
    for(int i=0; i<1001; i++){
		softmax_denom+=exp(mem_conv2[i]);
	}
	
	
	*prediction = 0;
	float max_score = 0;
	for(int i=0; i<1001; i++){
		//if score of current class is greater than the max score, set
		//predicted class to the current class
		if( (exp(mem_conv2[i])/softmax_denom) > max_score ){
			max_score = (exp(mem_conv2[i])/softmax_denom);
			*prediction = i;
			//printf("Prediction: %d\n", *prediction);
		}
	}
	
    //printf("Final Prediction: %d\n", *prediction);
}
