#ifndef LOGIT_LAYER_H
#define LOGIT_LAYER_H

void avg_pool2d(float* input, float* output, int O, int I, int N, int M, int K, int stride);
void dropout(float* input, int O, int I, int N, int M, int keep_prob);

#endif
