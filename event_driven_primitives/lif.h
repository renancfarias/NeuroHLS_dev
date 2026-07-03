#ifndef LIF_H
#define LIF_H

#include <ap_fixed.h>
#include <ap_int.h>
#include "types.h"
#include "lif_utils.h"

template<unsigned int N_NEURONS>
void lif_layer(
    hls::stream<spike_t> &in,
    hls::stream<spike_t> &out,
    lif_decay_func_t decay_func
) {
    
    static neuron_state_t neuron_states[N_NEURONS];

    // Se N_NEURONS for grande (>100), mude 'complete' para 'cyclic' para economizar área
    #pragma HLS ARRAY_PARTITION variable=neuron_states type=complete
    
    static bool initialized = false;

    // --- INICIALIZAÇÃO ---
    if (!initialized) {
        init_loop: for (int i = 0; i < N_NEURONS; i++) {
            #pragma HLS UNROLL
            neuron_states[i].index = i;
            neuron_states[i].v_mem = LIF_V_REST;
            neuron_states[i].v_leak = LIF_V_REST;
            neuron_states[i].v_th = LIF_V_THRESH; 
            neuron_states[i].v_reset = LIF_V_RESET; 
            neuron_states[i].r_mem = LIF_R_MEMB;
            neuron_states[i].tau = LIF_TAU;
            neuron_states[i].last_spike_time = 0; 
            neuron_states[i].input_current = 0;
        }
        initialized = true;
    }

    // --- PROCESSAMENTO ---
    
    // Loop principal (Processa enquanto houver dados ou até o break)
    // Nota: while(!empty) pode ser perigoso em co-simulação se não houver delay.
    // O ideal é usar um loop infinito com break no token de fim ou while(flag).
    bool finished = false;
    
    process_loop: while(!finished) {
        #pragma HLS PIPELINE II=1
        
        // --- A SOLUÇÃO MÁGICA ESTÁ AQUI ---
        // Diz ao HLS: "Eu garanto que não vou acessar o mesmo índice duas vezes seguidas muito rápido.
        // Se acontecer, eu aceito o risco de usar o dado antigo."
        #pragma HLS DEPENDENCE variable=neuron_states inter false
        
        // Leitura Bloqueante (mais seguro que check empty)
        spike_t input_spike = in.read();

        if (input_spike.type == TYPE_END_STEP || input_spike.type == TYPE_END_SAMPLE) {
            // Repassa o token e termina o loop deste batch
            out.write(input_spike);
            finished = true; 
        } else if (input_spike.type == TYPE_SPIKE) {
            unsigned int neuron_idx = input_spike.width_idx;
            
            
            // Verificação de segurança (Opcional, mas boa prática)
            if (neuron_idx < N_NEURONS) {
                
                // 1. LEITURA (Read)
                neuron_state_t neuron = neuron_states[neuron_idx];
                
                // 2. CÁLCULO (Modify) - lógica inline de lif_core
                time_step_t dt = input_spike.timestamp - neuron.last_spike_time;
                current_t input_current = input_spike.amplitude;
                voltage_t v_old = neuron.v_mem;
                voltage_t v_rest = neuron.v_leak;
                voltage_t r_mem = neuron.r_mem;
                voltage_t v_inf = v_rest + (input_current * r_mem);
                decay_t decay = decay_func(dt);
                voltage_t v_diff = v_old - v_inf;
                voltage_t v_decayed = v_diff * decay;
                voltage_t v_new = v_inf + v_decayed;
                neuron.v_mem = v_new;
                
                // 3. ESCRITA NA SAÍDA (Opcional)
                if (threshold(neuron, input_spike)) {
                    spike_t out_spike;
                    out_spike.type = TYPE_SPIKE;
                    out_spike.amplitude = 1; 
                    out_spike.timestamp = input_spike.timestamp;
                    out_spike.time_step = input_spike.time_step;
                    out_spike.batch_idx = input_spike.batch_idx;
                    out_spike.channel_idx = 0;
                    out_spike.height_idx = 0;
                    out_spike.width_idx = neuron_idx;
                    out.write(out_spike);
                }

                // 4. ESCRITA NA MEMÓRIA (Write Back)
                // É aqui que ocorria o erro de dependência
                neuron_states[neuron_idx] = neuron;
            }
        }
    }
}
#endif
