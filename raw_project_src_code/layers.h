#ifndef LAYERS_H
#define LAYERS_H

//  General layer input & output
unsigned char b[224][224];
unsigned char g[224][224];
unsigned char r[224][224];
float input[224][224][3];
float output[112][112][32];


///////////////////////////////////////////////////////////////
////////                  Used Layers                  ////////
///////////////////////////////////////////////////////////////
void perform_conv2d( float input[][224][3], float output[][112][32] );
void perform_BatchNorm ( float output[][112][32] );
void perform_Depthwise ( float output[][112][32] );

#endif