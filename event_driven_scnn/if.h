#ifndef LIF_H
#define LIF_H

#include <ap_fixed.h>
#include <ap_int.h>
#include "types.h"
#include <hls_stream.h>

// =========================================================
// 1. CONSTANTES E MACROS
// =========================================================

// Constantes Físicas
const voltage_t V_THRESH = 1.0;
const voltage_t V_RESET = 0.0;
const voltage_t R_MEMB = 1.0; 
const int RESET_INTERVAL_SAMPLES = 32;
const float BETA = 0.9;

// =========================================================
// 2. ESTRUTURAS DE DADOS INTERNAS
// =========================================================

/**
 * @brief Camada Integrate-and-Fire
 * @tparam IN_H Altura da Entrada
 * @tparam IN_W Largura da Entrada
 * @tparam IN_C Canais da Entrada
 * @tparam RESET_INTERVAL_STEPS Intervalo de steps para reset
 */
template<int IN_C, int IN_H, int IN_W>
void if_layer(hls::stream<spike_t> &in, 
    hls::stream<spike_t> &out, 
    voltage_t (&potentials)[IN_C][IN_H][IN_W]
) {
    bool end_of_processing = false;
    
    // 1. ESTADO PERSISTENTE (STATIC)
    #pragma HLS DEPENDENCE variable=potentials inter false

    #pragma HLS ARRAY_PARTITION variable=potentials complete dim=1 // Particiona totalmente a dimensão IN_C
    // #pragma HLS ARRAY_PARTITION variable=potentials complete dim=2 // Particiona totalmente a dimensão IN_H
    // #pragma HLS ARRAY_PARTITION variable=potentials complete dim=3 // Particiona totalmente a dimensão IN_W

    // static bool initialized = false;

    // // 2. INICIALIZAÇÃO
    // if (!initialized) {
    //     init_loop: for (int c = 0; c < IN_C; c++) {
    //         for (int h = 0; h < IN_H; h++) {
    //             for (int w = 0; w < IN_W; w++) {
    //                 #pragma HLS PIPELINE II=1
    //                 potentials[c][h][w] = V_RESET;
    //             }
    //         }
    //     }
    //     initialized = true;
    // }

    // 3. PROCESSAMENTO
    processing_loop: while (!end_of_processing) {
        
        spike_t s = in.read();

        if (s.type == TYPE_SPIKE) {
            
            // Acumula a corrente de entrada para o step atual
            ap_uint<16> c = s.channel_idx;
            ap_uint<16> h = s.height_idx;
            ap_uint<16> w = s.width_idx;
            
            // current_buffer[s.channel_idx][s.height_idx][s.width_idx] += s.amplitude;
            potentials[c][h][w] += s.amplitude * R_MEMB;

            if (potentials[c][h][w] >= V_THRESH) {
                spike_t out_spike = s;
                out_spike.type = TYPE_SPIKE;
                out_spike.amplitude = 1.0;
                out.write(out_spike); // Saída

                potentials[c][h][w] = V_RESET; // Reset Imediato
            }
        }
        else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            if (s.type == TYPE_END_SAMPLE) {
                reset_loop_w:
                for (int w = 0; w < IN_W; w++) {
                    reset_loop_h:
                    for (int h = 0; h < IN_H; h++) {
                        #pragma HLS PIPELINE II=1
                        reset_loop_c:
                        for (int c = 0; c < IN_C; c++) {
                            potentials[c][h][w] = V_RESET; // Reset Completo no final da amostra
                        }
                    }
                }
            }
            out.write(s); // Propaga o sinal de controle (END_STEP ou END_BATCH)
            end_of_processing = true; // Para este exemplo, processamos apenas um step. Remova esta linha para processamento contínuo.
        }
    }
    
}
#endif
