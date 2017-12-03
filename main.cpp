#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>


#include "loadbmp.h"
#include "layers.h"


//cast int -> string type
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

int main(int argc,char **argv){

    //load BMP RGB array into input array
    load_bmp(argc, argv, b, g, r);
	int mean = 0;
	int std = 255;
	
    for(int x = 0; x<224; x++){
        for(int y = 0; y < 224; y++){
            input[x][y][0] = ((float)r[x][y]-mean)/std;
            input[x][y][1] = ((float)g[x][y]-mean)/std;
            input[x][y][2] = ((float)b[x][y]-mean)/std;
            
        }
    }
	
	perform_conv2d( input, output );
	perform_BatchNorm( output );
	perform_Depthwise( output );
    
    return 0;
}

