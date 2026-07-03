#ifndef LINEAR_H
#define LINEAR_H

#include <ap_fixed.h>
#include "types.h"


template <int NUM_INPUTS, int NUM_OUTPUTS, int NP>
void linear_layer(
    hls::stream<spike_t> &input, 
    hls::stream<spike_t> &output, 
    const weight_t weights[NUM_OUTPUTS][NUM_INPUTS], // const ajuda o compilador a inferir ROM
    const weight_t biases[NUM_OUTPUTS]
) { 
    // -------------------------------------------------------------------------
    // 1. OTIMIZAÇÃO DE MEMÓRIA E PARALELISMO
    // -------------------------------------------------------------------------
    
    // Buffer local (Registradores/BRAM)
    accum_t acc_buffer[NUM_OUTPUTS];
    
    // Permite acessar 'NP' linhas da matriz de pesos ao mesmo tempo
    #pragma HLS ARRAY_PARTITION variable=weights type=cyclic factor=NP dim=1
    
    // Permite escrever em 'NP' acumuladores ao mesmo tempo
    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=complete

    // -------------------------------------------------------------------------
    // 2. INICIALIZAÇÃO (Bias)
    // -------------------------------------------------------------------------
    // Unroll factor=NP garante que fazemos isso em blocos, não 1 por 1
    init_loop: for (int i = 0; i < NUM_OUTPUTS; i++) {
        #pragma HLS pipeline ii=1
        #pragma HLS UNROLL factor=NP
        acc_buffer[i] = biases[i];
    }

    // Variáveis de estado
    bool is_last = false;
    time_step_t batch_time = 0; // Armazena o tempo do frame atual

    // -------------------------------------------------------------------------
    // 3. PROCESSAMENTO (Input Stationary)
    // -------------------------------------------------------------------------
    process_loop: while (!is_last) {
        #pragma HLS PIPELINE II=1
        
        // Leitura Bloqueante (Seguro para Hardware)
        spike_t s = input.read();

        if (s.last_feature) {
            is_last = true;
            batch_time = s.timestamp; // Captura o tempo correto do fim do frame
        } 
        else {
            // Se NP < NUM_OUTPUTS, o pipeline II vai aumentar proporcionalmente.
            // Ex: Se Outputs=128 e NP=16, II=8 (8 ciclos por spike de entrada).
            // Se NP=128, II=1 (1 ciclo por spike -> Máxima performance).
            
            ap_uint<16> in_idx = s.index;
            current_t amp = s.amplitude;

            update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS UNROLL factor=NP
                
                // Acessa a coluna 'in_idx' de todas as linhas 'out_idx'
                acc_buffer[out_idx] += weights[out_idx][in_idx] * amp;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. WRITE BACK (Saída)
    // -------------------------------------------------------------------------
    write_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
        #pragma HLS PIPELINE II=1
        
        current_t output_current = acc_buffer[out_idx];
        
        // Enviamos apenas correntes não-nulas para economizar tráfego
        // Nota: O LIF precisa saber lidar com ausência de input (apenas decaimento)
        if (output_current != 0) {
            spike_t out_spike;
            out_spike.index = out_idx;
            out_spike.amplitude = output_current;
            out_spike.timestamp = batch_time; // Usa o tempo capturado
            out_spike.last_feature = false;
            
            output.write(out_spike);
        }
    }

    // Token de Finalização
    spike_t end_token;
    end_token.index = 0;
    end_token.amplitude = 0;
    end_token.timestamp = batch_time;
    end_token.last_feature = true;
    output.write(end_token);
}


    
#endif