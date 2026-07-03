#ifndef NETWORK_H
#define NETWORK_H

#include <hls_stream.h>
#include "types.h"



template<int NUM_INPUTS>
inline void get_spikes(
    current_t input_spikes[NUM_INPUTS], 
    hls::stream<spike_t> &input_stream, 
    time_step_t current_time
) {
    // 1. Loop de Processamento (Envia apenas dados úteis)
    for (int i = 0; i < NUM_INPUTS; i++) {
        #pragma HLS PIPELINE II=1
        
        // Verifica se é zero (Esparsidade)
        if (input_spikes[i] != 0) {
            spike_t event;
            event.index = i;
            event.amplitude = input_spikes[i];
            event.timestamp = current_time;
            
            // DADOS NORMAIS SEMPRE TÊM LAST = FALSE
            event.last_feature = false; 
            
            input_stream.write(event);
        }
    }

    // 2. Token de Finalização (EOS - End of Stream)
    // Enviado OBRIGATORIAMENTE após terminar de varrer o vetor,
    // independente se o último dado era 0 ou não.
    spike_t end_token;
    end_token.index = 0;       // Irrelevante (dummy)
    end_token.amplitude = 0;   // Irrelevante (dummy)
    end_token.timestamp = current_time;
    end_token.last_feature = true; // <--- AQUI ESTÁ A CHAVE
    
    input_stream.write(end_token);
}

void snn(hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes);

void scnn (hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes);

#endif