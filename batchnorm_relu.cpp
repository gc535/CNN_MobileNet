#include <stdio.h>
#include <string.h>
#include <math.h>

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
	

	//RELU, will go through matrix elementwise and make negative values 0
	//float relu_output[O][O][N];

	for(int i=0; i<O; i++){
		for(int j=0; j<O; j++){
			for(int k=0; k<N; k++){
				int i_index = i + j * O + k * ofmap_size;
				input[i_index] = (input[i_index] > 0) ? input[i_index] : 0;
				//if(input[i_index]<0){
				//	input[i_index]=0;
				//}
		
			}
		}
	}

	return 0;
}
