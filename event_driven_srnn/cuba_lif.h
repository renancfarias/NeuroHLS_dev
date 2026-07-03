#ifndef CUBA_LIF_H
#define CUBA_LIF_H


#include <stdint.h>
#include <stdbool.h>
#include <math.h>
// Inclui o método de aproximação que criamos anteriormente
#include "ts_efa.h"
#include "exp_pwl_lut.h"
#include "types.h"

/**
 * @brief Assinatura do ponteiro de função para o cálculo de decaimento (exponencial).
 * 
 * Através deste ponteiro de função (`typedef`), podemos injetar qualquer
 * algoritmo de aproximação exponencial (TS-EFA, LUT, Taylor, etc)
 * sem precisar modificar a lógica interna de atualização do neurônio.
 * 
 * @param scaled_dt O delta_t já escalonado pelo fator da constante de tempo.
 * @return O fator multiplicador do decaimento em Fixed-Point (ex: Q16.16).
 */
typedef decay_t (*exp_decay_func_ptr)(accum_t exp_arg);

/**
 * @brief Atualiza os estados de um neurônio Current-Based (CUBA) Leaky Integrate-and-Fire.
 * 
 * @param v_mem Ponteiro para o potencial de membrana atual.
 * @param i_syn Ponteiro para a corrente sináptica atual.
 * @param dt Tempo decorrido desde a última atualização (evento).
 * @param input_weight Peso do spike de entrada que acabou de chegar.
 * @param tau_mem Constante de tempo da membrana (não escalada).
 * @param tau_syn Constante de tempo da sinapse (não escalada).
 * @param v_th Threshold (limiar) para o neurônio disparar (spike).
 * @param v_leak Potencial de repouso do neurônio.
 * @param v_reset Potencial de reset do neurônio.
 * @param exp_decay_func Ponteiro para a função de decaimento que desejamos usar.
 * @return Amplitude do spike emitido, ou 0 se não houve spike.
 */
inline current_t update_cuba_lif_neuron(
    voltage_t* v_mem,
    current_t* i_syn,
    time_step_t dt,
    current_t input_current,
    weight_t input_weight,
    weight_t r,
    tau_t tau_mem,
    tau_t tau_syn,
    voltage_t v_th,
    voltage_t v_leak,
    voltage_t v_reset,
    exp_decay_func_ptr exp_decay_func
) {
    #pragma HLS INLINE off
    const accum_t dt_f = (accum_t)dt;
    const accum_t tau_m = (accum_t)tau_mem;
    const accum_t tau_s = (accum_t)tau_syn;
    const accum_t v0 = (accum_t)(*v_mem);
    const accum_t i0 = (accum_t)(*i_syn);

    const accum_t decay_mem = (accum_t)exp_decay_func(-(dt_f / tau_m));
    const accum_t decay_syn = (accum_t)exp_decay_func(-(dt_f / tau_s));

    // Solução analítica entre eventos (S(t)=0)
    accum_t v_t = (accum_t)v_leak + (v0 - (accum_t)v_leak) * decay_mem;
    const accum_t tau_diff = tau_m - tau_s;
    if (tau_diff != (accum_t)0) {
        v_t += ((accum_t)r * i0 * tau_s / tau_diff) * (decay_mem - decay_syn);
    } else {
        // Caso degenerado: tau_mem == tau_syn == tau
        v_t += ((accum_t)r * i0 / tau_m) * dt_f * decay_mem;
    }

    accum_t i_t = i0 * decay_syn;

    // Salto instantâneo da corrente no instante do evento
    accum_t delta_i = (accum_t)input_weight * (accum_t)input_current;
    *i_syn = (current_t)(i_t + delta_i);

    // Verifica threshold antes da conversao para evitar wrap em voltage_t.
    if (v_t >= (accum_t)v_th) {
        *v_mem = v_reset;
        return (current_t)1;
    }

    *v_mem = (voltage_t)v_t;
    
    return (current_t)0; // Não emitiu Spike
}

/**
 * @brief Wrapper (atalho) para o CUBA LIF utilizando explicitamente a função TS-EFA.
 */
// inline bool update_cuba_lif_tsefa(
//     voltage_t* v_mem, 
//     current_t* i_syn, 
//     time_step_t dt, 
//     current_t input_current,
//     weight_t input_weight, 
//     tau_t tau_mem_scaled, 
//     tau_t tau_syn_scaled, 
//     voltage_t v_th
// ) {
//     // Injetamos a função `ts_efa_compute_decay` dinamicamente:
//     return update_cuba_lif_neuron(
//         v_mem, i_syn, dt, input_current, input_weight, 
//         tau_mem_scaled, tau_syn_scaled, v_th, 
//         ts_efa_compute_decay // <-- INJEÇÃO DO TS-EFA
//     );
// }


template<int IN_C, int IN_H, int IN_W>
void CubaLIF(hls::stream<spike_t> &in, 
    hls::stream<spike_t> &out,
    tau_t  tau_syn [IN_W],
    tau_t  tau_mem [IN_W],
    weight_t  r [IN_W],
    weight_t v_leak [IN_W],
    weight_t v_threshold [IN_W],
    weight_t v_reset [IN_W],
    weight_t w_in [IN_W]
){
    static voltage_t v_mem[IN_C][IN_H][IN_W] = {};
    static current_t i_syn[IN_C][IN_H][IN_W] = {};
    static current_t i_new[IN_C][IN_H][IN_W] = {};
    static time_step_t last_time;
    static time_step_t current_time;
    #pragma HLS ARRAY_PARTITION variable=v_mem type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=i_syn type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=i_new type=complete dim=3

    while (!in.empty()){
        spike_t s = in.read();
        
        
        if (s.type == TYPE_SPIKE) {
            uint16_t c = s.channel_idx;
            uint16_t h = s.height_idx;
            uint16_t w = s.width_idx;

            i_new[c][h][w] += s.amplitude; // Incrementa a corrente de entrada com o peso do spike
            
        } else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            current_time = s.timestamp;
            // Atualiza o estado de cada neurônio no final do passo de tempo
            for (int c = 0; c < IN_C; c++) {
                for (int h = 0; h < IN_H; h++) {
                    for (int w = 0; w < IN_W; w++) {
                        time_step_t dt = current_time - last_time;

                        current_t spike_amplitude = update_cuba_lif_neuron(
                            &v_mem[c][h][w],
                            &i_syn[c][h][w],
                            dt,
                            i_new[c][h][w], // Corrente acumulada dos spikes recebidos neste passo
                            w_in[w], // Peso de entrada para este neurônio
                            r[w], // Resistência
                            tau_mem[w],
                            tau_syn[w],
                            v_threshold[w],
                            v_leak[w], // Potencial de repouso
                            v_reset[w], // Potencial de reset
                            exp_pwl_lut_decay_fp // Usando a função de decaimento TS-EFA
                        );

                        if (spike_amplitude != (current_t)0) {
                            spike_t out_spk;
                            out_spk.type = TYPE_SPIKE;
                            out_spk.amplitude = spike_amplitude;
                            out_spk.timestamp = current_time;
                            out_spk.time_step = s.time_step;
                            out_spk.batch_idx = s.batch_idx;
                            out_spk.channel_idx = c;
                            out_spk.height_idx = h;
                            out_spk.width_idx = w;
                            out.write(out_spk);
                        }
                        

                        i_new[c][h][w] = 0; // Limpa a corrente de entrada para o próximo passo
                    }
                }
            }
            last_time = current_time; // Atualiza o tempo do último passo
            out.write(s); // Propaga o pacote de fim de passo ou fim de amostra para a saída
            if (s.type == TYPE_END_SAMPLE) {
                last_time = 0;
                for (int c = 0; c < IN_C; c++) {
                    for (int h = 0; h < IN_H; h++) {
                        for (int w = 0; w < IN_W; w++) {
                            v_mem[c][h][w] = v_reset[w]; // Reseta o potencial de membrana para o valor de reset
                            i_syn[c][h][w] = 0; // Limpa a corrente sináptica
                        }
                    }
                }
            }
            break;
        }
    }
}

#endif // CUBA_LIF_H
