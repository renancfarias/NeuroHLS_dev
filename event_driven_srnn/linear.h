#ifndef LINEAR_H
#define LINEAR_H

#include "types.h"
#include <hls_stream.h>

// Linear Layer (Fully Connected)
// Design: Stream Processing (consome um Time Step inteiro por chamada)
// Entrada: Já achatada pela função anterior. O índice linear vem em width_idx.
template <int NUM_INPUTS, int NUM_OUTPUTS>
void Linear(
    hls::stream<spike_t> &input_stream, 
    hls::stream<spike_t> &output_stream, 
    const weight_t weights[NUM_OUTPUTS][NUM_INPUTS]
) { 
    // 1. ESTADO PERSISTENTE
    static accum_t acc_buffer[NUM_OUTPUTS];

    // SOLUÇÃO: Particionar completamente array pequeno remove dependência de BRAM
    // Se NUM_OUTPUTS for pequeno (<= 64), complete é ideal. 
    // Se for grande, use cyclic e aceite II maior ou use DEPENDENCE.
    //#pragma HLS ARRAY_PARTITION variable=acc_buffer type=complete
    
    // Otimização de acesso aos pesos para casar com o unroll abaixo
    //#pragma HLS ARRAY_PARTITION variable=weights type=complete dim=1

    static bool initialized = false;

    // 2. INICIALIZAÇÃO
    if (!initialized) {
        init_v: for (int i = 0; i < NUM_OUTPUTS; i++) {
            #pragma HLS PIPELINE II=1
            acc_buffer[i] = 0.0;
        }
        initialized = true;
    }

    // 3. LOOP DE PROCESSAMENTO
    bool end_of_processing = false;
    
    main_loop: while (!end_of_processing) {
        // Pipeline II=1 no loop externo permite processar 1 spike por clock
        // #pragma HLS PIPELINE
        
        spike_t s = input_stream.read();

        if (s.type == TYPE_SPIKE) {
            unsigned int in_idx = s.width_idx;
            current_t amp = s.amplitude;

            if (in_idx < NUM_INPUTS) {
                // Ao desenrolar completamente o loop interno, fazemos todas as somas em paralelo.
                // Isso só é viável porque o array acc_buffer foi particionado (registros).
                update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                    #pragma HLS PIPELINE
                    acc_buffer[out_idx] += weights[out_idx][in_idx] * amp;
                }
            }
        }
        else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            // Emissão de saídas
            output_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS PIPELINE
                
                accum_t potential = acc_buffer[out_idx];
                
                
                spike_t out_spike;
                out_spike.type = TYPE_SPIKE;
                out_spike.amplitude = (current_t)potential;
                out_spike.channel_idx = 0;
                out_spike.height_idx = 0;
                out_spike.width_idx = out_idx;
                out_spike.timestamp = s.timestamp;
                out_spike.time_step = s.time_step;
                out_spike.batch_idx = s.batch_idx;
                output_stream.write(out_spike); // Escrita serial

                acc_buffer[out_idx] = 0.0; // Reseta o acumulador
                   
            }

            output_stream.write(s);
            
            
            end_of_processing = true;
            
        }
    }
}

template <int NUM_INPUTS, int NUM_OUTPUTS>
void linear_layer_pruning(
    hls::stream<spike_t> &input_stream, 
    hls::stream<spike_t> &output_stream, 
    const weight_t weights[NUM_OUTPUTS][NUM_INPUTS], 
    const weight_t biases[NUM_OUTPUTS],
    const weight_t tol
) { 
    // 1. ESTADO PERSISTENTE
    static accum_t acc_buffer[NUM_OUTPUTS];

    // SOLUÇÃO: Particionar completamente array pequeno remove dependência de BRAM
    // Se NUM_OUTPUTS for pequeno (<= 64), complete é ideal. 
    // Se for grande, use cyclic e aceite II maior ou use DEPENDENCE.
    //#pragma HLS ARRAY_PARTITION variable=acc_buffer type=complete
    
    // Otimização de acesso aos pesos para casar com o unroll abaixo
    //#pragma HLS ARRAY_PARTITION variable=weights type=complete dim=1

    static bool initialized = false;

    static int valid_weight_count[NUM_INPUTS] = {0}; // number of valid weights per input
    static int valid_weight_indices[NUM_INPUTS][NUM_OUTPUTS]; // to keep track of original indices of valid weights

    // 2. INICIALIZAÇÃO
    if (!initialized) {
        // faz a poda de pesos zero aqui para evitar multiplicações desnecessárias no loop de processamento
        prune_loop: for (int in_idx = 0; in_idx < NUM_INPUTS; in_idx++) {
            for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS PIPELINE II=1
                if (weights[out_idx][in_idx] != 0 && (weights[out_idx][in_idx] > tol || weights[out_idx][in_idx] < -tol)) {
                    valid_weight_indices[in_idx][valid_weight_count[in_idx]] = out_idx; // Armazena o índice original do peso válido
                    valid_weight_count[in_idx]++;   
                }
            }
        }

        init_v: for (int i = 0; i < NUM_OUTPUTS; i++) {
            #pragma HLS PIPELINE II=1
            acc_buffer[i] = biases[i]; 
        }
    

        initialized = true;
    }

    // 3. LOOP DE PROCESSAMENTO
    bool end_of_processing = false;
    
    while (!input_stream.empty()) {
        // Pipeline II=1 no loop externo permite processar 1 spike por clock
        // #pragma HLS PIPELINE
        
        spike_t s = input_stream.read();

        if (s.type == TYPE_SPIKE) {
            unsigned int in_idx = s.width_idx;
            current_t amp = s.amplitude;

            if (in_idx < NUM_INPUTS) {
                // Ao desenrolar completamente o loop interno, fazemos todas as somas em paralelo.
                // Isso só é viável porque o array acc_buffer foi particionado (registros).
                update_loop: for (int out_idx = 0; out_idx < valid_weight_count[in_idx]; out_idx++) {
                    #pragma HLS PIPELINE
                    int original_out_idx = valid_weight_indices[in_idx][out_idx]; // Recupera o índice original do peso válido
                    acc_buffer[original_out_idx] += weights[original_out_idx][in_idx] * amp;
                }
            }
        }
        else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            // Emissão de saídas
            output_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS PIPELINE
                
                accum_t potential = acc_buffer[out_idx];
                
                if (potential !=0 ) {
                    spike_t out_spike;
                    out_spike.type = TYPE_SPIKE;
                    out_spike.amplitude = (current_t)potential;
                    out_spike.channel_idx = 0;
                    out_spike.height_idx = 0;
                    out_spike.width_idx = out_idx;
                    out_spike.timestamp = s.timestamp;
                    out_spike.time_step = s.time_step;
                    out_spike.batch_idx = s.batch_idx;
                    output_stream.write(out_spike); // Escrita serial
                }

                acc_buffer[out_idx] = biases[out_idx]; // Reseta o acumulador
            

                   
            }

            spike_t end_token = s;
            end_token.amplitude = 0;
            end_token.width_idx = 0; 
            output_stream.write(end_token);
            
           
            end_of_processing = true;
            
        }
    }
}
#endif


