This is the source code for the MobileNet c++ implementation. This sources code integrates the current pre-trained checkpoints from google. 
We included a Makefile in the "src" foler. In order to execute the code, you need to "cd" into "/src" and generate the executable by running 

$ make

To clean up the project (.o files and executable "Main"), type:

$ make clean

The executable will be generated in the root directory of this project. To test the inference, we provided four exmple pictures in the "/pic" directory. After generating the executable "Main" in the root directory, you can run the code by typing:

$ ./Main ./pic/<picture you want to test>

It will output the prediction label <1-1000>. We also included a label list in the "/params" called "label.txt". You can verify the prediction by comparing with this file.

