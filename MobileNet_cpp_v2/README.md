This version can run several test instance in batch.
In order to run the source code, you need to "make" the executable first: 
$ make
Then run the exectable main by:
$ main [number of tests in one batch] [mean] [stdv]  (mean = 0, and stdv = 255 for now)
the image path is hard coded in mobilenets.cpp file. You should change that accordingly in order to run your test images in a specific folder. All the images should be named as 0.bmp, 1.bmp, so on. (IT HAS TO START WITH 0!)
This version serves just as a runtime analysis. 

NOTE: It takes roughly 5-7 seconds to process a image!
