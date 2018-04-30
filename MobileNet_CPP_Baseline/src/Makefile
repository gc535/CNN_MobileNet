IDIR = ../lib
ODIR = ./obj

CXX = g++
CXXFLAGS = -g -Wall -I$(IDIR)
TARGET = Main



OBJ = loadbmp.o main.o perform_conv2d.o batchnorm_relu.o logit_layer.o
SRC = loadbmp.cpp main.cpp perform_conv2d.cpp batchnorm_relu.cpp logit_layer.cpp

#$(ODIR)/%.o : $(SRC)
	

$(TARGET) : $(OBJ)
	$(CXX) $(CXXFLAGS) -c -o $(ODIR)/%.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o ../$(TARGET) $(OBJ)


.PHONY: clean
	
clean: 
	rm -f *.o ../Main