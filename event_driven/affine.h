#ifndef AFFINE_H
#define AFFINE_H

#include "types.h"
#include <hls_stream.h>

// Definição de tipos locais (caso não estejam no types.h)
typedef ap_fixed<16, 8> weight_t;
typedef ap_fixed<32, 16> accum_t;

/**
 * @brief Camada Affine (Linear sem Bias) - Input Stationary
 * Calcula y = W * x
 * * @tparam NUM_INPUTS   Número de neurônios de entrada
 * @tparam NUM_OUTPUTS  Número de neurônios de saída
 * @tparam NP           Fator de Paralelismo (Quantos neurônios de saída atualizar por ciclo)
 */
template <int NUM_INPUTS, int NUM_OUTPUTS, int NP>
void affine_layer(
    hls::stream<spike_t> &input_stream, 
    hls::stream<spike_t> &output_stream, 
    const weight_t weights[NUM_OUTPUTS][NUM_INPUTS]
) { 
    // -------------------------------------------------------------------------
    // 1. MEMÓRIA E PARTICIONAMENTO
    // -------------------------------------------------------------------------
    static accum_t acc_buffer[NUM_OUTPUTS];

    // Particionamento para permitir acesso paralelo a 'NP' acumuladores/pesos
    #pragma HLS ARRAY_PARTITION variable=weights type=cyclic factor=NP dim=1
    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=cyclic factor=NP

    // -------------------------------------------------------------------------
    // 2. INICIALIZAÇÃO (ZERAR ACUMULADOR)
    // -------------------------------------------------------------------------
    // Diferença chave: Como não tem bias, iniciamos tudo com zero.
    init_loop: for (int i = 0; i < NUM_OUTPUTS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=NP
        acc_buffer[i] = 0;
    }

    bool is_last = false;
    time_step_t batch_time = 0;

    // -------------------------------------------------------------------------
    // 3. PROCESSAMENTO (Input Stationary)
    // -------------------------------------------------------------------------
    process_loop: while (!is_last) {
        #pragma HLS PIPELINE II=1
        
        spike_t s = input_stream.read();

        if (s.last_feature) {
            is_last = true;
            batch_time = s.timestamp;
        } 
        else {
            ap_uint<16> in_idx = s.index;
            current_t amp = s.amplitude;

            // Projeção: O spike de entrada atualiza 'NP' neurônios de saída simultaneamente
            update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS UNROLL factor=NP
                acc_buffer[out_idx] += weights[out_idx][in_idx] * amp;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. WRITE BACK (Disparo)
    // -------------------------------------------------------------------------
    write_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
        #pragma HLS PIPELINE II=1
        
        accum_t potential = acc_buffer[out_idx];
        
        if (potential != 0) {
            spike_t out_spike;
            out_spike.index = out_idx;
            out_spike.amplitude = potential;
            out_spike.timestamp = batch_time;
            out_spike.last_feature = false;
            
            output_stream.write(out_spike);
        }
    }

    // Token de Finalização
    spike_t end_token;
    end_token.index = 0;
    end_token.amplitude = 0;
    end_token.timestamp = batch_time;
    end_token.last_feature = true;
    
    output_stream.write(end_token);
}

#endif