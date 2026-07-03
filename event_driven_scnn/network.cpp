#include "linear.h"
#include "if.h"
// #include "lif.h"
// #include "lif_utils.h"
#include "conv2d.h"
#include "sumpool.h"
#include "types.h"
#include "nir_weights.h"
#include "network.h"
#include "debug_utils.h"
#include "flatten.h"

// Define a profundidade segura para absorver bursts das camadas Conv/Linear
#define FIFO_DEPTH 16

void scnn(hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes) {
   // #pragma HLS INTERFACE ap_ctrl_chain port=return
    #pragma HLS DATAFLOW disable_start_propagation
    
    // Tenta executar uma iteração de cada camada por ciclo de chamada do scnn
    // Se não houver dados em alguma camada, ela retorna imediatamente (graças ao input.empty() check)
    //#pragma HLS PIPELINE II=1

    // Declaração de Streams Estáticos 
    // Precisam ser static para manter os dados entre as chamadas da função scnn
    static hls::stream<spike_t> s0("s0"); 
    #pragma HLS STREAM variable=s0 depth=FIFO_DEPTH 

    static hls::stream<spike_t> s1("s1"); 
    #pragma HLS STREAM variable=s1 depth=FIFO_DEPTH

    static hls::stream<spike_t> s2("s2"); 
    #pragma HLS STREAM variable=s2 depth=FIFO_DEPTH

    static hls::stream<spike_t> s3("s3"); 
    #pragma HLS STREAM variable=s3 depth=FIFO_DEPTH

    static hls::stream<spike_t> s4("s4"); 
    #pragma HLS STREAM variable=s4 depth=FIFO_DEPTH

    static hls::stream<spike_t> s5("s5"); 
    #pragma HLS STREAM variable=s5 depth=FIFO_DEPTH

    static hls::stream<spike_t> s6("s6"); 
    #pragma HLS STREAM variable=s6 depth=FIFO_DEPTH
    
    static hls::stream<spike_t> s7("s7"); 
    #pragma HLS STREAM variable=s7 depth=FIFO_DEPTH

    static hls::stream<spike_t> s8("s8"); 
    #pragma HLS STREAM variable=s8 depth=FIFO_DEPTH

    static hls::stream<spike_t> s9("s9"); 
    #pragma HLS STREAM variable=s9 depth=FIFO_DEPTH
    
    static hls::stream<spike_t> s10("s10"); 
    #pragma HLS STREAM variable=s10 depth=FIFO_DEPTH
    
    static hls::stream<spike_t> s11("s11"); 
    #pragma HLS STREAM variable=s11 depth=FIFO_DEPTH

    static hls::stream<spike_t> sm("sm"); 
    #pragma HLS STREAM variable=sm depth=FIFO_DEPTH

    static voltage_t potentials_1[16][16][16] = {0};

    static voltage_t potentials_3[16][16][16] = {0};

    static voltage_t potentials_6[8][8][8] = {0};

    static voltage_t potentials_10[1][1][256] = {0};

    static voltage_t potentials_12[1][1][10] = {0};
    

    // --- EXECUÇÃO EM CADEIA ---
    // Chamamos cada camada uma vez. 
    // Se houver dados no stream de entrada dela, ela processa 1 pacote.
    // Se o stream anterior gerou um burst (ex: 100 spikes), esta cadeia precisará ser 
    // chamada 100 vezes pelo testbench para esvaziar os buffers.

    // L0: Conv
    conv2d_layer_no_bias<34, 34, 2, 16, 5, 5, 2, 2, 1, 1>(input_spikes, s0, weights_1);

    // L1: LIF
    if_layer<16, 16, 16>(s0, s1, potentials_1);

    // L2: Conv
    conv2d_layer_no_bias<16, 16, 16, 16, 3, 3, 1, 1, 1, 1>(s1, s2, weights_3);

    // // // Monitor
    // // monitor_layer(s2, s2_m, "hls_log.txt");

    // L3: LIF
    if_layer<16, 16, 16>(s2, s3, potentials_3);

    // // Monitor
    // monitor_layer(sm, s3, "hls_log.txt");

    //L4: Pool
    sumpool2d_layer<16, 16, 16, 2>(s3, s4);

    // L5: Conv
    conv2d_layer_no_bias<8, 8, 16, 8, 3, 3, 1, 1, 1, 1>(s4, s5, weights_6);

    // L6: LIF
    if_layer<8, 8, 8>(s5, s6, potentials_6);

    // L7: Pool
    sumpool2d_layer<8, 8, 8, 2>(s6, s7);

    // l8: Flatten
    flatten_layer<8, 4, 4>(s7, s8);

    // L9: Linear
    linear_layer<128, 256>(s8, s9, weights_10);

    // L10: LIF
    if_layer<1,1,256>(s9, s10, potentials_10);

    // L11: Linear
    linear_layer<256, 10>(s10, s11, weights_12);

    // L12: LIF Final
    if_layer<1,1,10>(s11, output_spikes, potentials_12);
}