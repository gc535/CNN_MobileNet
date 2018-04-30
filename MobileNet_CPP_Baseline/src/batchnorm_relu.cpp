/////////////////////////////////////////////////////////////////////////////////
// This code belongs to "Energy Efficient Deep Learning” project group at      //
// Cornell University, Graduate School of Electrical and Computer Engineering. //
// This code is strictly for research and education. Do not use it for any     //
// commercial purposes.                                                        //
// Author: Jonathan Wu; Guangwei Chen                                          //
/////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "main.h"


//REMOVE ALL UNNECESSARY DECLARATIONS AFTER 1D ARRAY IS WORKING

int perform_BatchNorm (float input[MAX_FMAP_SIZE], const float y, const float moving_variance[MAX_MOVING_VARIANCE], const float gamma[MAX_GAMMA], 
						const float moving_mean[MAX_MOVING_MEAN], const float beta[MAX_BETA], int N, int O){
// int perform_BatchNorm (const float y, const float moving_variance[MAX_MOVING_VARIANCE], const float gamma[MAX_GAMMA], 
// 						const float moving_mean[MAX_MOVING_MEAN], const float beta[MAX_BETA], int N, int O){

	float post_add_mv[N];
	float rec_sqrt[N];
	int ofmap_size = O*O;

	for(int ele=0; ele<N; ele++){
		post_add_mv[ele] = moving_variance[ele] + y;
		rec_sqrt[ele] = 1/sqrt(post_add_mv[ele]);
	}

	float mul_output[N];
	//float mul_output_1[O][O][N];


	for(int ele=0; ele<N; ele++){
		mul_output[ele] = rec_sqrt[ele] * gamma[ele];
		for(int i=0; i<O; i++){
			for(int j=0; j<O; j++){

				//1D implementation
				int i_index = i + j * O + ele * ofmap_size;
				input[i_index] = input[i_index] * mul_output[ele];

				//3D implementation
				//mul_output_1[i][j][ele] = output[i][j][ele] * mul_output[ele];
			}
		} 
	}
	/*
	//print values for checking
	printf("pixel 0, 0\n");
	fprintf(stderr, "Mul Node\n");
	//Check mul node values
	for(int i=0; i<N; i++){
		fprintf(stderr, "%f\n", mul_output[i]);
	}
	printf("\n");	
	fprintf(stderr, "Input multiplied w/ mul node\n");
	//post multiply node
	for(int i=0; i<N; i++){
		int o_index = 0 + 0 * O + i * (O*O);
		fprintf(stderr, "%f\n", input[o_index]);
	}	
	printf("\n");
	
	//print values for checking
	printf("pixel 111, 111\n");
	printf("\n");	
	fprintf(stderr, "Input multiplied w/ mul node\n");
	//post multiply node
	for(int i=0; i<N; i++){
		int o_index = 111 + 111 * O + i * (O*O);
		fprintf(stderr, "%f\n", input[o_index]);
	}	
	printf("\n");
	
	*/
	
	float mul_output_2[N];

	for(int ele=0; ele<N; ele++){
		mul_output_2[ele] = mul_output[ele] * moving_mean[ele];

	}


	float sub[N];

	for(int ele=0; ele<N; ele++){;
		sub[ele] = beta[ele] - mul_output_2[ele];
		for(int i=0; i<O; i++){
			for(int j=0; j<O; j++){

				//1D implementation
				int i_index = i + j * O + ele * ofmap_size;
				input[i_index] = input[i_index] + sub[ele];

				//3D implementation
				//add[i][j][ele] = mul_output_1[i][j][ele] + sub[ele];
			}
		}
	}
	/*
	//print values for checking 
	fprintf(stderr, "sub Node\n");
	//Check mul node values
	for(int i=0; i<N; i++){
		fprintf(stderr, "%f\n", sub[i]);
	}	
	printf("\n");
	printf("printing pixel 0,0\n");
	fprintf(stderr, "Input added w/ sub node\n");
	//post multiply node
	for(int i=0; i<N; i++){
		int o_index = O-1 + O-1 * O + i * (O*O);
		fprintf(stderr, "%f\n", input[o_index]);
	}
	
	
	//print values for checking 
	printf("\n");
	printf("checking pixel 111,111\n");
	fprintf(stderr, "Input added w/ sub node\n");
	//post multiply node
	for(int i=0; i<N; i++){
		int o_index = 111 + 111 * O + i * (O*O);
		fprintf(stderr, "%f\n", input[o_index]);
	}
	
	*/
	
	//RELU, will go through matrix elementwise and make negative values 0
	//float relu_output[O][O][N];

	for(int i=0; i<O; i++){
		for(int j=0; j<O; j++){
			for(int k=0; k<N; k++){
				int i_index = i + j * O + k * ofmap_size;
				input[i_index] = (input[i_index] > 0) ? input[i_index] : 0;
				input[i_index] = (input[i_index] <= 6) ? input[i_index] : 6;
				//if(input[i_index]<0){
				//	input[i_index]=0;
				//}
		
			}
		}
	}

	return 0;
}
