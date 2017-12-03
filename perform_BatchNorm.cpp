#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>


void perform_BatchNorm ( float output[][112][32] ){
    #include "data.h"
	///////////////////////////////////////////////////////////////
	////////          Batch Norm Layer parameters          ////////
	///////////////////////////////////////////////////////////////
	//pre mul
	float y = 0.0010000000475;   //value from tensorflow graph
	float post_add_mv[32];
	float rec_sqrt[32];

	//mul
	float mul_output[32];

	//mul_1
	float mul_output_1[112][112][32];

	//mul_2
	float mul_output_2[32];

	//post mul_2
	float sub[32];
	float add[112][112][32];


	//PERFORM BATCHNORM 
    //pre mul
	for(int ele=0; ele<32; ele++){
		post_add_mv[ele] = moving_variance_0[ele] + y;
		rec_sqrt[ele] = 1/sqrt(post_add_mv[ele]);
	}

    
    //mul & mul_1
	for(int ele=0; ele<32; ele++){
		mul_output[ele] = rec_sqrt[ele] * gamma_0[ele];
		for(int i=0; i<112; i++){
			for(int j=0; j<112; j++){
				mul_output_1[i][j][ele] = output[i][j][ele] * mul_output[ele];
			}
		} 
	}

    
    //mul_2
	for(int ele=0; ele<32; ele++){
		mul_output_2[ele] = mul_output[ele] * moving_mean_0[ele];

	}


    //post mul_2
	for(int ele=0; ele<32; ele++){;
		sub[ele] = beta_0[ele] - mul_output_2[ele];
		for(int i=0; i<112; i++){
			for(int j=0; j<112; j++){
				add[i][j][ele] = mul_output_1[i][j][ele] + sub[ele];
			}
		}
	}
	

	//RELU, will go through matrix elementwise and make negative values 0
	//float relu_output[112][112][32];
	for(int i=0; i<112; i++){
		for(int j=0; j<112; j++){
			for(int k=0; k<32; k++){
				if(add[i][j][k]<0){
					output[i][j][k]=0;
				}else{
					output[i][j][k] = add[i][j][k];
				}
			}
		}
	}
}