#include "linear.h"
#include "lif.h"
#include "conv2d.h"
#include "sumpool.h"
#include "n-mnist_weights.h"
#include "nir_weights.h"
#include "network.h"
    
void snn(hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes) {

#pragma HLS DATAFLOW

// --- STREAMS (CANAIS) ---
// Defina a profundidade (depth) para evitar stalls se houver bursts de spikes
#pragma HLS STREAM variable=input_spikes 

hls::stream<spike_t> output_linear_0("s_l0");
#pragma HLS STREAM variable=output_linear_0 

hls::stream<spike_t> output_lif_0("s_lif0");
#pragma HLS STREAM variable=output_lif_0 

hls::stream<spike_t> output_linear_1("s_l1");
#pragma HLS STREAM variable=output_linear_1 

hls::stream<spike_t> output_lif_1("s_out");
#pragma HLS STREAM variable=output_lif_1 

// --- VARIÁVEIS ESTÁTICAS (PESOS) ---
static weight_t biases_0[128] = {0};
static weight_t biases_1[10] = {0};

// --- TAREFAS (TASKS) ---

// 1. Camadas Ocultas
linear_layer<784, 128, 128>(input_spikes, output_linear_0, weights_0, biases_0);
lif_layer<128>(output_linear_0, output_lif_0);

// 2. Camadas de Saída
linear_layer<128, 10, 10>(output_lif_0, output_linear_1, weights_1, biases_1);
lif_layer<10>(output_linear_1, output_spikes);

}

void scnn (hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes) {

    #pragma HLS DATAFLOW
    #pragma HLS INTERFACE ap_ctrl_chain port=return
    
    // --- STREAMS (CANAIS) ---
    // Definindo profundidade para evitar stalls no Dataflow
    #pragma HLS STREAM variable=input_spikes depth=16
    #pragma HLS STREAM variable=output_spikes depth=16

    // Declaração de todos os streams intermediários
    hls::stream<spike_t> output_0("s_0");
    #pragma HLS STREAM variable=output_0 depth=16

    hls::stream<spike_t> output_1("s_1");
    #pragma HLS STREAM variable=output_1 depth=16

    hls::stream<spike_t> output_2("s_2");
    #pragma HLS STREAM variable=output_2 depth=16

    hls::stream<spike_t> output_3("s_3");
    #pragma HLS STREAM variable=output_3 depth=16

    hls::stream<spike_t> output_4("s_4");
    #pragma HLS STREAM variable=output_4 depth=16

    hls::stream<spike_t> output_5("s_5");
    #pragma HLS STREAM variable=output_5 depth=16

    hls::stream<spike_t> output_6("s_6");
    #pragma HLS STREAM variable=output_6 depth=16

    hls::stream<spike_t> output_7("s_7");
    #pragma HLS STREAM variable=output_7 depth=16

    // Streams para as camadas densas
    hls::stream<spike_t> output_9("s_9");
    #pragma HLS STREAM variable=output_9 depth=16

    hls::stream<spike_t> output_10("s_10");
    #pragma HLS STREAM variable=output_10 depth=16

    hls::stream<spike_t> output_11("s_11");
    #pragma HLS STREAM variable=output_11 depth=16


    // ====================================================
    // FEATURE EXTRACTION (CONVOLUTIONAL LAYERS)
    // ====================================================

    // Layer 0: Conv2D Input 34x34x2 -> Output 16x16x16
    // Formula: (34 - 5 + 2*1)/2 + 1 = 16
    conv2d_layer<34, 34, 2, 16, 5, 5, 2, 2, 1, 1>
        (input_spikes, output_0, layer_0_weight, layer_0_bias);

    // Layer 1: LIF Activation
    // Neurons: 16 * 16 * 16 = 4096
    if_layer<4096>
        (output_0, output_1);

    // Layer 2: conv2d_layer Input 17x17x16 -> Output 16x16x16
    // Formula: (16 - 3 + 2*1)/1 + 1 = 16 (Mantém tamanho, altera profundidade se necessario)
    conv2d_layer<16, 16, 16, 16, 3, 3, 1, 1, 1, 1>
        (output_1, output_2, layer_2_weight, layer_2_bias);

    // Layer 3: LIF Activation
    // Neurons: 16 * 16 * 16 = 4096
    if_layer<4096>
        (output_2, output_3);

    // Layer 4: SumPool Input 17x17 -> Output 8x8 (Integer Division)
    // Formula: 17 / 2 = 8
    sumpool2d_layer<16, 16, 16, 2, 8, 8>
        (output_3, output_4);

    // Layer 5: Conv2D Input 8x8x16 -> Output 8x8x8
    // Formula: (8 - 3 + 2*1)/1 + 1 = 8
    conv2d_layer<8, 8, 16, 8, 3, 3, 1, 1, 1, 1>
        (output_4, output_5, layer_5_weight, layer_5_bias);

    // Layer 6: LIF Activation
    // Neurons: 8 * 8 * 8 = 512
    if_layer<512>
        (output_5, output_6);

    // Layer 7: sumpool_layer Input 8x8 -> Output 4x4
    // Formula: 8 / 2 = 4
    sumpool2d_layer<8, 8, 8, 2, 4, 4>
        (output_6, output_7);

    // ====================================================
    // CLASSIFICATION (DENSE LAYERS)
    // ====================================================

    // Layer 8: Flattening (Implícito)
    // A saída do Pool (4x4x8) resulta em 128 features.
    // Conectamos output_7 diretamente na entrada da linear_layer.

    // Layer 9: Linear 128 -> 256
    // Entrada calculada: 4 * 4 * 8 = 128
    linear_layer<128, 256, 16>
        (output_7, output_9, layer_9_weight, layer_9_bias);

    // Layer 10: LIF Activation
    if_layer<256>
        (output_9, output_10);

    // Layer 11: Linear 256 -> 10
    linear_layer<256, 10, 10>
        (output_10, output_11, layer_11_weight, layer_11_bias);

    // Layer 12: LIF Final Activation
    if_layer<10>
        (output_11, output_spikes);
}
