include ../Makefile.inc

CPATH=/opt/xilinx/Xilinx_SDx_2017.2_sdx_0823_1/SDx/2017.2/Vivado_HLS/include/
CFLAGS:=-I../minizip -I../utils $(CFLAGS) -I$(CPATH)
LDFLAGS:=-L../utils -lCraftUtils $(LDFLAGS)

OBJ=Mobilenets.o
#OLD_EXE=quantize_params_kh.exe quantize_params_nb.exe
EXE=Mobilenets.exe

all: $(EXE)

$(OBJ): %.o: %.cpp %.h
	$(CXX) -c $< -o $@ $(CFLAGS)
	
%.o: %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for executables
$(EXE): %.exe: %.o $(OBJ)
	g++ $^ -o $@ $(CFLAGS) $(LDFLAGS) 
clean:
	rm *.exe *.o

#test:
#	g++ -o test $(CFLAGS) $(LDFLAGS) -std=gnu++11 test_weight_load.cpp
