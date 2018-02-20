#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>


#include "main.h"
#include "loadbmp.h"
#include "perform_conv2d.h"
#include "batchnorm_relu.h"
#include "logit_layer.h"

unsigned char* readBMP(char*);

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


unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

<<<<<<< HEAD
int main(int argc,char **argv){
    /*
=======
    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    for(i = 0; i < size; i += 3)
    {
            unsigned char tmp = data[i];
            data[i] = data[i+2];
            data[i+2] = tmp;
    }
	//data should contain the (R, G, B) values of the pixels. The color of pixel (i, j) is stored at 
	//data[j * width + i], data[j * width + i + 1] and data[j * width + i + 2].
    return data;
}



int main(int argc,char **argv){
	
	unsigned char *raw_RGB_input = readBMP(argv[1]);
	
>>>>>>> 7718e49381efb12b35cfb463c47ddb2fc376f7a4
	float mem_conv1[MAX_FMAP_SIZE] = {
		#include "normalized_car2_bmp.dat"
		};
	//float mem_conv1[MAX_FMAP_SIZE];
  	float mem_conv2[MAX_FMAP_SIZE];
  	Layer_Helper lh;
<<<<<<< HEAD
  	*/
=======
  	
  	//~ for(int i=0; i<224; i++){
		//~ for(int j=0; j<224; j++){
			//~ int i_index_r = i + j*224 + 0*(224*224);
			//~ int i_index_g = i + j*224 + 1*(224*224);
			//~ int i_index_b = i + j*224 + 2*(224*224);
			//~ float r_val = atof( (const char*) raw_RGB_input[i + j*224]);
			//~ float g_val = atof( (const char*) raw_RGB_input[i + j*224 + 1]);
			//~ float b_val = atof( (const char*) raw_RGB_input[i + j*224 + 2]);
			//~ mem_conv1[i_index_r] = (float) ( (r_val-mean)/st_dev );
			//~ mem_conv1[i_index_g] = (float) ( (g_val-mean)/st_dev );
			//~ mem_conv1[i_index_b] = (float) ( (b_val-mean)/st_dev );
		//~ }
	//~ }
  	
  	/*
>>>>>>> 7718e49381efb12b35cfb463c47ddb2fc376f7a4
	unsigned char b_input[224][224];
	unsigned char g_input[224][224];
	unsigned char r_input[224][224];

    float mem_conv1[MAX_FMAP_SIZE];
    float mem_conv2[MAX_FMAP_SIZE];

    //load BMP RGB array into input array
    load_bmp(argc, argv, b_input, g_input, r_input);
    //printf("red = %d, green = %d, blue = %d. \n", r_input[0][0], g_input[0][0], b_input[0][0]);

    
    for(int x = 0; x<224; x++){
        for(int y = 0; y < 224; y++){

        	//1D array mapping(new implementation)
        	int i_index_r = x + y*224 + 0*(224*224);
        	int i_index_g = x + y*224 + 1*(224*224);
        	int i_index_b = x + y*224 + 2*(224*224);
        	mem_conv1[i_index_r] = ((float)r_input[x][y]-mean)/st_dev;
        	mem_conv1[i_index_g] = ((float)g_input[x][y]-mean)/st_dev;
        	mem_conv1[i_index_b] = ((float)b_input[x][y]-mean)/st_dev;
        	//printf("X: %d\t Y: %d\n", x, y);
			//printf("R: %d, G:%d, B:%d\n", mem_conv1[i_index_r], mem_conv1[i_index_g], mem_conv1[i_index_b]);
        	//3D array mapping(working)
            // input[x][y][0] = ((float)r[x][y]-mean)/st_dev;
            // input[x][y][1] = ((float)g[x][y]-mean)/st_dev;
            // input[x][y][2] = ((float)b[x][y]-mean)/st_dev;
            
        }
    }
    
	

	//CONVOLUTION 
	//int O = 112; //output feature map widthbmp_normalized_from_python.dat
	//int I = 224; //input feature map width
	//int N=32; //number of output feature maps
	//int M=3; //number of input feature maps
    //int K=3; //kernel width
    //int S=2; //stride

    //void perform_conv2d (float* input, float* output, const float* conv_weight, int O, int I, int N, int M, int K, int S);
	//int perform_BatchNorm (float* input_bn, const float y, const float* moving_variance, const float* gamma, const float* moving_mean, const float* beta, int N, int O);

	//fprintf(stderr, "layer 0\n");
  	//layer 0
	perform_conv2d(mem_conv1, mem_conv2, conv_weight_0, 112, 224, 32, 3, 3, 2);
	
	//for(int i=0; i<32; i++){
	//	int base_offset = 111 + 111 * 112 + i * (112*112);
	//	fprintf(stderr, "%e\n", mem_conv2[base_offset]);
	//}
    //lh.get_intermediate(mem_conv2, 112, 32);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_0, gamma_0, moving_mean_0, beta_0, 32, 112);
    
	//for(int i=0; i<32; i++){
	//	int base_offset = 111 + 111 * 112 + i * (112*112);
	//	fprintf(stderr, "%e\n", mem_conv2[base_offset]);
	//}

	//fprintf(stderr, "layer 1 depthwise\n");
	
	
	//layer 1 depthwise
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_1, 112, 112, 1, 32, 3, 1);
	//for(int i=0; i<32; i++){
	//	int base_offset = 0 + 0 * 112 + i * (112*112);
	//	fprintf(stderr, "%e\n", mem_conv1[base_offset]);
	//}
	
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_1, gamma_depth_1, moving_mean_depth_1, beta_depth_1, 32, 112); //on pixel 111,111, channel 26 and 3, values are double what we expect 
	//1, 4 channel 30
	//for(int i=0; i<32; i++){
	//	int base_offset = 111 + 111 * 112 + i * (112*112);
	//	fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//}
	

 	//fprintf(stderr, "layer 1 pointwise\n");
 	
 	
	//layer 1 pointwise
	//void perform_conv2d (float* input, float* output, const float* conv_weight, int O, int I, int N, int M, int K, int S);
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_1, 112, 112, 64, 32, 1, 1);

	//printf("checking pixel 0,0\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 0 + 0 * 112 + i * (112*112);
	//	fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//}
	//printf("checking pixel 111,111\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 111 + 111 * 112 + i * (112*112);
	//	fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//}


	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_1, gamma_point_1, moving_mean_point_1, beta_point_1, 64, 112);
	//printf("checking pixel 111,111\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 111 + 111 * 112 + i * (112*112);
	//	fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//}
	//printf("checking pixel 0,0\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 0 + 0 * 112 + i * (112*112);
	//	fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//}

	//layer2 depthwise
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_2, 56, 112, 1, 64, 3, 2);
	

	

	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_2, gamma_depth_2, moving_mean_depth_2, beta_depth_2, 64, 56);
	//printf("checking pixel 55,55\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 55 + 55 * 56 + i * (56*56);
	//	fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//}
	//printf("checking pixel 0,0\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 0 + 0 * 56 + i * (56*56);
	//	fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//}

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
	//printf("checking pixel 27,27\n");
	//for(int i=0; i<128; i++){
	//	int base_offset = 27 + 27 * 28 + i * (28*28);
	//	fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//}
	//printf("checking pixel 0,0\n");
	//for(int i=0; i<64; i++){
	//	int base_offset = 0 + 0 * 28 + i * (28*28);
	//	fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//}

	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_4, gamma_depth_4, moving_mean_depth_4, beta_depth_4, 128, 28);


	//layer4 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_4, 28, 28, 256, 128, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_4, gamma_point_4, moving_mean_point_4, beta_point_4, 256, 28);	
	
	//~ printf("checking pixel 27,27\n");
	//~ for(int i=0; i<128; i++){
		//~ int base_offset = 27 + 27 * 28 + i * (28*28);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<64; i++){
		//~ int base_offset = 0 + 0 * 28 + i * (28*28);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }

	//layer5 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_5, 28, 28, 1, 256, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_5, gamma_depth_5, moving_mean_depth_5, beta_depth_5, 256, 28);	
	
	//~ printf("checking pixel 27,27\n");
	//~ for(int i=0; i<256; i++){
		//~ int base_offset = 27 + 27 * 28 + i * (28*28);
		//~ fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<256; i++){
		//~ int base_offset = 0 + 0 * 28 + i * (28*28);
		//~ fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//~ }	
	

	//layer5 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_5, 28, 28, 256, 256, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_5, gamma_point_5, moving_mean_point_5, beta_point_5, 256, 28);	
	
	//~ printf("checking pixel 27,27\n");
	//~ for(int i=0; i<256; i++){
		//~ int base_offset = 27 + 27 * 28 + i * (28*28);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<256; i++){
		//~ int base_offset = 0 + 0 * 28 + i * (28*28);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }	

		
	//layer6 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_6, 14, 28, 1, 256, 3, 2);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_6, gamma_depth_6, moving_mean_depth_6, beta_depth_6, 256, 14);	

	//~ printf("checking pixel 13,13\n");
	//~ for(int i=0; i<256; i++){
		//~ int base_offset = 13 + 13 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<256; i++){
		//~ int base_offset = 0 + 0 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//~ }

	//layer6 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_6, 14, 14, 512, 256, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_6, gamma_point_6, moving_mean_point_6, beta_point_6, 512, 14);	

		
	//layer7 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_7, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_7, gamma_depth_7, moving_mean_depth_7, beta_depth_7, 512, 14);
		
	//layer7 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_7, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_7, gamma_point_7, moving_mean_point_7, beta_point_7, 512, 14);	
	
	//~ printf("checking pixel 13,13\n");
	//~ for(int i=0; i<512; i++){
		//~ int base_offset = 13 + 13 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<512; i++){
		//~ int base_offset = 0 + 0 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }

		
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

	//~ printf("checking pixel 13,13\n");
	//~ for(int i=0; i<512; i++){
		//~ int base_offset = 13 + 13 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<512; i++){
		//~ int base_offset = 0 + 0 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	
			
	//layer10 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_10, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_10, gamma_depth_10, moving_mean_depth_10, beta_depth_10, 512, 14);
		
	//layer10 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_10, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_10, gamma_point_10, moving_mean_point_10, beta_point_10, 512, 14);
	
	//~ printf("checking pixel 13,13\n");
	//~ for(int i=0; i<64; i++){
		//~ int base_offset = 13 + 13 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<64; i++){
		//~ int base_offset = 0 + 0 * 14 + i * (14*14);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }

	//layer11 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_11, 14, 14, 1, 512, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_11, gamma_depth_11, moving_mean_depth_11, beta_depth_11, 512, 14);
		
	//layer11 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_11, 14, 14, 512, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_11, gamma_point_11, moving_mean_point_11, beta_point_11, 512, 14);
			
	//layer12 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_12, 7, 14, 1, 512, 3, 2);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_12, gamma_depth_12, moving_mean_depth_12, beta_depth_12, 512, 7);
	
	//~ printf("checking pixel 6,6\n");
	//~ for(int i=0; i<512; i++){
		//~ int base_offset = 6 + 6 * 7 + i * (7*7);
		//~ fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<512; i++){
		//~ int base_offset = 0 + 0 * 7 + i * (7*7);
		//~ fprintf(stderr, "%f\n", mem_conv1[base_offset]);
	//~ }	
	
		
	//layer12 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_12, 7, 7, 1024, 512, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_12, gamma_point_12, moving_mean_point_12, beta_point_12, 1024, 7);
	
	//~ printf("checking pixel 6,6\n");
	//~ for(int i=0; i<1024; i++){
		//~ int base_offset = 6 + 6 * 7 + i * (7*7);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<1024; i++){
		//~ int base_offset = 0 + 0 * 7 + i * (7*7);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }	
		
	

			
	//layer13 depth
	perform_depthwise_conv2d(mem_conv2, mem_conv1, depth_conv_weight_13, 7, 7, 1, 1024, 3, 1);
	perform_BatchNorm(mem_conv1, 0.0010000000475, moving_variance_depth_13, gamma_depth_13, moving_mean_depth_13, beta_depth_13, 1024, 7);
		
	//layer13 point
	perform_pointwise_conv2d(mem_conv1, mem_conv2, point_conv_weight_13, 7, 7, 1024, 1024, 1, 1);
	perform_BatchNorm(mem_conv2, 0.0010000000475, moving_variance_point_13, gamma_point_13, moving_mean_point_13, beta_point_13, 1024, 7);
	
	//~ printf("checking pixel 6,6\n");
	//~ for(int i=0; i<1024; i++){
		//~ int base_offset = 6 + 6 * 7 + i * (7*7);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }
	//~ printf("checking pixel 0,0\n");
	//~ for(int i=0; i<1024; i++){
		//~ int base_offset = 0 + 0 * 7 + i * (7*7);
		//~ fprintf(stderr, "%f\n", mem_conv2[base_offset]);
	//~ }	

  
    //LOGIT LAYER
    //void perform_conv2d (float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], const float conv_weight[MAX_W_CONV], int O, int I, int N, int M, int K, int S){
	//void avg_pool2d(float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], int O, int I, int N, int M, int K, int S)				 
    avg_pool2d(mem_conv2, mem_conv1, 1, 7, 1024, 1024, 7, 2);
    
    
    //~ printf("avgpooling check\n");
	//~ for(int i=0; i<1024; i++){
		//~ fprintf(stderr, "%f\n", mem_conv1[i]);
	//~ }	
    
    //dropout operation not needed in inference, only training
    //dropout(mem_conv1, 1, 1, 1024, 1024, 50);
    perform_pointwise_conv2d(mem_conv1, mem_conv2, logit_conv_weight, 1, 1, 1001, 1024, 1, 1);
    
    //~ printf("conv2d check\n");
	//~ for(int i=0; i<1001; i++){
		//~ fprintf(stderr, "%f\n", mem_conv2[i]);
	//~ }	

 
    //Add Biases
    for(int i=0; i<1001; i++){
		mem_conv2[i] += logit_bias_weight[i];
	}

    printf("logit bias check\n");
	for(int i=0; i<1001; i++){
		fprintf(stderr, "%f\n", mem_conv2[i]);
	}	
 
    //Predictions
    float softmax_denom = 0;
    for(int i=0; i<1001; i++){
		softmax_denom+=exp(mem_conv2[i]);
	}
	
	
	int prediction = 0;
	float max_score = 0;
	for(int i=0; i<1001; i++){
		//if score of current class is greater than the max score, set
		//predicted class to the current class
		if( (exp(mem_conv2[i])/softmax_denom) > max_score ){
			max_score = (exp(mem_conv2[i])/softmax_denom);
			prediction = i;
			printf("Prediction: %d\n", prediction);
		}
	}
	
    
    //To Do: 
    // - Softmax activation
    // - Return predicted class
    //		- Create mapping for classes based on labels.txt
    // - Test Inference on images
    
    printf("Final Prediction: %d\n", prediction);
 
    return 0;
}
