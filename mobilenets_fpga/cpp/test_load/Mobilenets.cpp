//------------------------------------------------------------------------
// Classifies a series of images
//------------------------------------------------------------------------
#include <iostream>

#include "ParamIO.h"
//#include "DataIO.h"
#include "Timer.h"
//#include "Common.h"
#include "Mobilenets.h"



//extern ostream cout;
float conv_weights[3*3*3*32];

//extern ostream cout;
void load_weights(const float* w){
	for(int i=0; i<3*3*3*32; i++){
		conv_weights[i] = w[i];
	}
}

int main(){
  // Quantize and load params to layers
  Params params(get_root_dir() + params_file);
  load_weights(params.float_data(0));
  for(int i=0; i<32; i++){
	  //iterate through each output channels 0,0 kernel for input channel 0
	  int w_index = 0 + 0*3 + (i*3+0) * 9;
	  //cout << conv_weights[w_index];
	  printf("%f\n", conv_weights[w_index]);
	  
  }



  return 0;
}
