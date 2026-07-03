#ifndef AFFINE_H
#define AFFINE_H

#include "types.h"
#include <hls_stream.h>


/**
 * @brief Camada Affine (Linear sem Bias) - Input Stationary
 * Suporta TYPE_END_BATCH para reset de contagem no testbench.
 */
template <int IN_H, int IN_W, int IN_C, int NUM_OUTPUTS, int NP>
void affine_layer(
    hls::stream<spike_t> &input_stream, 
    hls::stream<spike_t> &output_stream, 
    const weight_t weights[NUM_OUTPUTS][IN_H * IN_W * IN_C]
) { 
    // Calcula o tamanho total da entrada linearizada
    constexpr int NUM_INPUTS = IN_H * IN_W * IN_C;

    // 1. MEMÓRIA E PARTICIONAMENTO
    static accum_t acc_buffer[NUM_OUTPUTS];

    #pragma HLS ARRAY_PARTITION variable=weights type=cyclic factor=NP dim=1
    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=cyclic factor=NP

    // 2. INICIALIZAÇÃO (ZERAR ACUMULADOR)
    init_loop: for (int i = 0; i < NUM_OUTPUTS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=NP
        acc_buffer[i] = 0;
    }

    bool is_last = false;
    
    // Variável para capturar qual token parou o loop (End Step ou End Batch)
    char termination_type = TYPE_END_STEP;

    // Cache de Metadados
    time_step_t cached_timestamp = 0;
    ap_uint<16> cached_time_step = 0;
    ap_uint<16> cached_batch_idx = 0;

    // 3. PROCESSAMENTO
    process_loop: while (!is_last) {
        //#pragma HLS PIPELINE II=1
        
        spike_t s = input_stream.read();

        // Verifica flag de fim (Passo ou Batch)
        if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            is_last = true;
            
            // --- CAPTURA DO TIPO ---
            termination_type = s.type;

            // Preserva metadados do token final
            cached_timestamp = s.timestamp;
            cached_time_step = s.time_step;
            cached_batch_idx = s.batch_idx;
        } 
        else if (s.type == TYPE_SPIKE) {
            // Atualiza cache de metadados
            cached_timestamp = s.timestamp;
            cached_time_step = s.time_step;
            cached_batch_idx = s.batch_idx;

            // Flattening On-the-fly
            int in_idx = ((s.height_idx * IN_W) + s.width_idx) * IN_C + s.channel_idx;
            
            current_t amp = s.amplitude;

            if (in_idx < NUM_INPUTS) {
                update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                    #pragma HLS UNROLL factor=NP
                    acc_buffer[out_idx] += weights[out_idx][in_idx] * amp;
                }
            }
        }
    }

    // 4. WRITE BACK (Disparo)
    write_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
        #pragma HLS PIPELINE II=1
        
        accum_t potential = acc_buffer[out_idx];
        
        if (potential != 0) {
            spike_t out_spike;
            
            out_spike.type = TYPE_SPIKE;
            out_spike.amplitude = potential;
            
            // Mapeamento de Saída 1D
            out_spike.channel_idx = out_idx;
            out_spike.height_idx = 0;
            out_spike.width_idx = 0;

            out_spike.timestamp = cached_timestamp;
            out_spike.time_step = cached_time_step;
            out_spike.batch_idx = cached_batch_idx;
            
            output_stream.write(out_spike);
        }
    }

    // 5. TOKEN DE FINALIZAÇÃO (Propagação)
    spike_t end_token;
    
    // Usa o tipo capturado
    end_token.type = termination_type;
    
    end_token.amplitude = 0;
    end_token.channel_idx = 0;
    end_token.height_idx = 0;
    end_token.width_idx = 0;

    end_token.timestamp = cached_timestamp;
    end_token.time_step = cached_time_step;
    end_token.batch_idx = cached_batch_idx;
    
    output_stream.write(end_token);
}

#endif