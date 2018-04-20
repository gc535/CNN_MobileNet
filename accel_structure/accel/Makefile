IDIR = ./
ODIR = ./

CXX = g++
CXXFLAGS = -g -Wall -I$(IDIR)
TARGET = main



OBJ = loadbmp.o mobilenets.o conv_layers.o logit_layer.o
SRC = loadbmp.cpp mobilenets.cpp conv_layers.cpp logit_layer.cpp

#$(ODIR)/%.o : $(SRC)
	

$(TARGET) : $(OBJ)
	#$(CXX) $(CXXFLAGS) -c -o $(ODIR)/%.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o ./$(TARGET) $(OBJ)


.PHONY: clean
	
clean: 
	rm -f main *.o 