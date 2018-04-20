#include <assert.h>
#include <ap_axi_sdata.h>
typedef ap_axiu<32,4,5,5> AXI_VAL;
template <typename T, int DIM, int SIZE, int U, int TI, int TD>
void wrapped_mmult_hw(AXI_VAL in_stream[2*SIZE], AXI_VAL out_stream[SIZE])
{
    T A[DIM][DIM], B[DIM][DIM], C[DIM][DIM];
    assert(sizeof(T)*8 == 32);
    // stream in the 2 input matrices
    for(int i=0; i<DIM; i++)
    for(int j=0; j<DIM; j++)
    {
        #pragma HLS PIPELINE II=1
        int k = i*DIM + j;
        A[i][j] = pop_stream<T,U,TI,TD>(in_stream[k]);
    }
    for (int i=0; i<DIM; i++)
    for (int j=0; j<DIM; j++)
    {
        #pragma HLS PIPELINE II=1
        int k = i*DIM + j + SIZE;
        B[i][j] = pop_stream<T,U,TI,TD>(in_stream[k]);
    }
    // do multiplication
    mmult_hw<T, DIM>(A, B, C);
    // stream out result matrix
    for (int i=0; i<DIM; i++)
    for (int j=0; j<DIM; j++)
    {
        #pragma HLS PIPELINE II=1
        int k = i*DIM + j;
        out_stream[k] = push_stream<T,U,TI,TD>(C[i][j], k==1023);
    }
}
// this is the top level design that will be synthesized into RTL
void HLS_accel(AXI_VAL INPUT_STREAM[2048], AXI_VAL OUTPUT_STREAM[1024])
{
    // Map ports to Vivado HLS interfaces
    #pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS
    #pragma HLS INTERFACE axis port=INPUT_STREAM
    #pragma HLS INTERFACE axis port=OUTPUT_STREAM
    wrapped_mmult_hw<float,32,32*32,4,5,5>(INPUT_STREAM, OUTPUT_STREAM);
}