/////////////////////////////////////////////////////////////////////////////////
// This code belongs to "Energy Efficient Deep Learning” project group at      //
// Cornell University, Graduate School of Electrical and Computer Engineering. //
// This code is strictly for research and education. Do not use it for any     //
// commercial purposes.                                                        //
// Author: Jonathan Wu; Guangwei Chen                                          //
/////////////////////////////////////////////////////////////////////////////////
#ifndef LOGIT_LAYER_H
#define LOGIT_LAYER_H

void avg_pool2d(float* input, float* output, int O, int I, int N, int M, int K, int stride);
void dropout(float* input, int O, int I, int N, int M, int keep_prob);

#endif
