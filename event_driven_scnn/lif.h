// #ifndef LIF_H
// #define LIF_H

// #include <ap_fixed.h>
// #include <ap_int.h>
// #include "types.h"
// #include "lif_utils.h"

// template<int IN_C, int IN_H, int IN_W>
// void lif_layer(hls::stream<spike_t> &in, hls::stream<spike_t> &out) {
    
//     static neuron_state_t neuron_states[IN_C][IN_H][IN_W];

//     // Se N_NEURONS for grande (>100), mude 'complete' para 'cyclic' para economizar área
//     #pragma HLS ARRAY_PARTITION variable=neuron_states type=complete
    
//     static bool initialized = false;

//     // --- INICIALIZAÇÃO ---
//     if (!initialized) {
//         init_loop: for (int c = 0; c < IN_C; c++) {
//             for (int h = 0; h < IN_H; h++) {
//                 for (int w = 0; w < IN_W; w++) {
//                     #pragma HLS UNROLL
//                     neuron_states[c][h][w].v_mem = V_REST;
//                     neuron_states[c][h][w].v_leak = V_REST;
//                     neuron_states[c][h][w].v_th = V_THRESH; 
//                     neuron_states[c][h][w].v_reset = V_RESET; 
//                     neuron_states[c][h][w].r_mem = R_MEMB;
//                     neuron_states[c][h][w].tau = TAU;
//                     neuron_states[c][h][w].last_spike_time = 0; 
//                     neuron_states[c][h][w].input_current = 0;
//             }
//         }
//     }
//         initialized = true;
//     }

//     // --- PROCESSAMENTO ---
    
//     // Loop principal (Processa enquanto houver dados ou até o break)
//     // Nota: while(!empty) pode ser perigoso em co-simulação se não houver delay.
//     // O ideal é usar um loop infinito com break no last_feature ou while(flag).
//     bool finished = false;
    
//     process_loop: while(true) {
//         #pragma HLS PIPELINE II=1
        
//         spike_t s = in.read();

//         if (s.type == TYPE_SPIKE) {
//             // --- PROCESSA SPIKE ---
//             neuron_state_t neuron = neuron_states[s.channel_idx][s.height_idx][s.width_idx];
//             // Atualiza o potencial de membrana com o novo input
//             lif_core(neuron, s);
//             // Verifica se o neurônio dispara
//             if (neuron.v_mem >= neuron.v_th) {
//                 // Soft Reset
//                 neuron.v_mem = neuron.v_mem - neuron.v_th;
//                 neuron.last_spike_time = s.timestamp; // Atualiza o tempo do último spike
//                 // Gera spike de saída
//                 spike_t out_spike = s;
//                 out_spike.type = TYPE_SPIKE;
//                 out_spike.amplitude = 1; 
//                 out.write(out_spike);
//             }

//             // Write Back
//             neuron_states[s.channel_idx][s.height_idx][s.width_idx] = neuron;
   
//         }
//         else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
//             // --- PROPAGAÇÃO DE CONTROLE ---
//             if (s.type == TYPE_END_SAMPLE) {
//                 initialized = false; // reset os potencias das membranas
//             }
//             out.write(s);
//             break; // Sai do loop de processamento
//         }
                


//     }
// }


// #endif