#ifndef LIF_H
#define LIF_H

#include <ap_fixed.h>
#include <ap_int.h>
#include "types.h"
#include "lif_utils.h"

template<unsigned int N_NEURONS>
void lif_layer(hls::stream<spike_t> &in, hls::stream<spike_t> &out) {
    
    static neuron_state_t neuron_states[N_NEURONS];

    // Se N_NEURONS for grande (>100), mude 'complete' para 'cyclic' para economizar área
    #pragma HLS ARRAY_PARTITION variable=neuron_states type=complete
    
    static bool initialized = false;

    // --- INICIALIZAÇÃO ---
    if (!initialized) {
        init_loop: for (int i = 0; i < N_NEURONS; i++) {
            #pragma HLS UNROLL
            neuron_states[i].index = i;
            neuron_states[i].v_mem = V_REST;
            neuron_states[i].v_leak = V_REST;
            neuron_states[i].v_th = V_THRESH; 
            neuron_states[i].v_reset = V_RESET; 
            neuron_states[i].r_mem = R_MEMB;
            neuron_states[i].tau = TAU;
            neuron_states[i].last_spike_time = 0; 
            neuron_states[i].input_current = 0;
        }
        initialized = true;
    }

    // --- PROCESSAMENTO ---
    
    // Loop principal (Processa enquanto houver dados ou até o break)
    // Nota: while(!empty) pode ser perigoso em co-simulação se não houver delay.
    // O ideal é usar um loop infinito com break no last_feature ou while(flag).
    bool finished = false;
    
    process_loop: while(!finished) {
        #pragma HLS PIPELINE II=1
        
        // --- A SOLUÇÃO MÁGICA ESTÁ AQUI ---
        // Diz ao HLS: "Eu garanto que não vou acessar o mesmo índice duas vezes seguidas muito rápido.
        // Se acontecer, eu aceito o risco de usar o dado antigo."
        #pragma HLS DEPENDENCE variable=neuron_states inter false
        
        // Leitura Bloqueante (mais seguro que check empty)
        spike_t input_spike = in.read();

        if (input_spike.last_feature) {
            // Repassa o token e termina o loop deste batch
            out.write(input_spike);
            finished = true; 
        } 
        else {
            unsigned int neuron_idx = input_spike.index;
            
            
            // Verificação de segurança (Opcional, mas boa prática)
            if (neuron_idx < N_NEURONS) {
                
                // 1. LEITURA (Read)
                neuron_state_t neuron = neuron_states[neuron_idx];
                
                // 2. CÁLCULO (Modify) - Demora vários ciclos (float latency)
                lif_core(neuron, input_spike);
                
                // 3. ESCRITA NA SAÍDA (Opcional)
                if (threshold(neuron, input_spike)) {
                    spike_t out_spike;
                    out_spike.index = neuron_idx;
                    out_spike.amplitude = 1; 
                    out_spike.timestamp = input_spike.timestamp;
                    out_spike.last_feature = false;
                    out.write(out_spike);
                }

                // 4. ESCRITA NA MEMÓRIA (Write Back)
                // É aqui que ocorria o erro de dependência
                neuron_states[neuron_idx] = neuron;
            }
        }
    }
}


/**
 * @brief Camada Integrate-and-Fire Sequencial
 * @tparam N_NEURONS Número de neurônios na camada
 */
template<unsigned int N_NEURONS>
void if_layer(hls::stream<spike_t> &in, hls::stream<spike_t> &out) {
    
    static neuron_state_t neuron_states[N_NEURONS];
    
    // Particionamento cíclico se você planeja desenrolar loops
    #pragma HLS ARRAY_PARTITION variable=neuron_states type=cyclic factor=4

    static bool initialized = false;

    // --- INICIALIZAÇÃO ---
    if (!initialized) {
        for (int i = 0; i < N_NEURONS; i++) {
            #pragma HLS PIPELINE II=1
            neuron_states[i].index = i;
            neuron_states[i].v_mem = 0.0;       // V_REST geralmente é 0
            neuron_states[i].v_th = V_THRESH; 
            neuron_states[i].v_reset = V_RESET; 
            neuron_states[i].last_spike_time = 0.0; 
        }
        initialized = true;
    }

    // --- PROCESSAMENTO ---
    bool finished = false;
    
    process_loop: while(!finished) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=neuron_states inter false

        spike_t input_spike = in.read();

        if (input_spike.last_feature) {
            out.write(input_spike); // Propaga token
            finished = true; 
        } 
        else {
            unsigned int neuron_idx = input_spike.index;
            
            if (neuron_idx < N_NEURONS) {
                // 1. Leitura
                neuron_state_t neuron = neuron_states[neuron_idx];
                
                // 2. Núcleo IF (Apenas soma)
                if_core(neuron, input_spike);
                
                // 3. Verifica Disparo
                if (threshold_check(neuron, input_spike)) {
                    spike_t out_spike;
                    out_spike.index = neuron_idx;
                    out_spike.amplitude = 1; 
                    out_spike.timestamp = input_spike.timestamp;
                    out_spike.last_feature = false;
                    out.write(out_spike);
                }

                // 4. Escrita
                neuron_states[neuron_idx] = neuron;
            }
        }
    }
}

#endif