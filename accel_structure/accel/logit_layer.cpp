#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//#include "mobilenets.h"
#include "mobilenets_updated.h"


//REMOVE ALL UNNECESSARY DECLARATIONS AFTER 1D ARRAY IS WORKING

void avg_pool2d(float input[MAX_FMAP_SIZE], float output[MAX_FMAP_SIZE], int O, int I, int N, int M, int K, int S)
{
	// initialize output fmaps,	
	for(int i = 0; i < MAX_FMAP_SIZE; i++) output[i] = 0;
	
	int ifmap_size = I*I;
	int ofmap_size = O*O;

	// perform convolution 
	for (int n = 0; n < N; n++) {//#iterate over number of output channels
			//accumulator variable gets incremented by each pixel value in input fmap
			float accumulator = 0;
			for (int x = 0; x < I; x++) {//iterate over number of rows in output fmap
				for (int y = 0; y < I; y++) {//iterate over number of cols in output fmap
					int i_index = x + y*I + n*ifmap_size;
					accumulator += input[i_index];
				}
			}
			int o_index = n * ofmap_size;
			//set output to average of input fmap
			output[o_index] = (accumulator/ifmap_size);

	}
}

/*  Function: dropout(float[], int, int, int)
    Argument: input feature map[I] * M,
              output feature map[O] * N.
              keep_prob: probablity to drop out each element in the matrix

    Describption: With probability `keep_prob`, outputs the input element scaled up by
                  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
                  sum is unchanged.                                             
*/
void dropout(float input[MAX_FMAP_SIZE], int O, int I, int N, int M, int keep_prob )
{
    int count = 0;
    for(int i=0; i < M; i++){
        if(rand()%100 < keep_prob) input[i] = 0; //value dropout for possibility under keep_prob 
        count++;
    } 
    for(int i=0; i < N; i++) input[i] = input[i] * (O/count);  //scale the matrix by 1/keep_prob, so that the sum is unchannged
}
