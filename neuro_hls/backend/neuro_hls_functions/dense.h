#pragma once

#include "bit_type.h"

// template<int n_neurons, int unroll_factor, typename potential_type>
// void dense_LIF(potential_type potentials[n_neurons], bit_t output[n_neurons])
// {
//     #pragma HLS INLINE
//     #pragma HLS BIND_OP variable=potentials op=mul impl=dsp

//     leaky_fire_dense_apply_decay:
//     for (int n = 0; n < n_neurons; n++)
//     {
//         #pragma HLS UNROLL factor=unroll_factor
//         potentials[n] *= layer::decay;
//     }

//     leaky_fire_dense_check_threshold:
//     for (int n = 0; n < n_neurons; n++)
//     {
//         #pragma HLS UNROLL factor=unroll_factor

//         if (potentials[n] >= layer::threshold)
//         {
//             output[n] = 1;
//             potentials[n] -= layer::threshold; ///// POR ENQUANTO, SUPORTE APENAS PARA SUBTRACT EM CASO DE FIRE
//         }
//         else
//         {
//             output[n] = 0;
//         }
//     }
// }

template<typename input_type, int size> void Merge(input_type (&receiver)[size], input_type (&other)[size])
{
    for (int i = 0; i < size; i++)
    {
        receiver[i] += other[i];
    }
}

template<typename input_type, int channels, int width> void Merge(input_type (&receiver)[channels][width], input_type (&other)[channels][width])
{
    for (int ch = 0; ch < channels; ch++)
    {
        for (int w = 0; w < width; w++)
        {
            receiver[ch][w] += other[ch][w];
        }
    }
}

template<typename input_type, int channels, int height, int width> void Merge(input_type (&receiver)[channels][height][width], input_type (&other)[channels][height][width])
{
    for (int ch = 0; ch < channels; ch++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                receiver[ch][h][w] += other[ch][h][w];
            }
        }
    }
}

template<
    int unroll_factor,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void multiply_and_accumulate(input_type (&input)[n_inputs],
                             result_type (&result)[n_neurons],
                             params_type (&weights)[n_neurons][n_inputs])
{
    result_type aux[unroll_factor];

    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    // #pragma HLS ARRAY_PARTITION variable=aux factor=unroll_factor dim=1 type=cyclic

    mult_and_accum_inputs:
    for (int i = 0; i < n_inputs; i += unroll_factor)
    {
    // #pragma HLS PIPELINE off
        
        mult_and_accum_neurons:
        for (int n = 0; n < n_neurons; n++)
        {
            // #pragma HLS PIPELINE off

            mult_and_accum_batch:
            for (int k = 0; k < unroll_factor; k++)
            {
                // #pragma HLS UNROLL

                aux[k] = weights[n][i + k] * input[i + k];
                result[n] += aux[k];
            }
        }
    }
};

template<
    int unroll_factor,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void Linear(input_type (&input)[n_inputs],
           result_type (&result)[n_neurons],
           params_type (&weights)[n_neurons][n_inputs])
{
    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    
    linear_bias:
    for (int n = 0; n < n_neurons; n++)
    {
        // #pragma HLS UNROLL factor=unroll_factor
        result[n] = 0;
    }

    multiply_and_accumulate<unroll_factor>(input, result, weights);
};

template<
    int unroll_factor,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void Affine(input_type (&input)[n_inputs],
           result_type (&result)[n_neurons],
           params_type (&weights)[n_neurons][n_inputs],
           params_type (&bias)[n_neurons])
{
    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    
    affine_bias:
    for (int n = 0; n < n_neurons; n++)
    {
        // #pragma HLS UNROLL factor=unroll_factor
        result[n] = bias[n];
    }

    multiply_and_accumulate<unroll_factor>(input, result, weights);
};

/**
 * SumPooling HLS Paramétrico
 * --------------------------------------------
 * T:    Tipo de dado (int, float, ap_fixed)
 * IN_H: Altura da entrada
 * IN_W: Largura da entrada
 * K_H:  Altura do Kernel (Janela)
 * K_W:  Largura do Kernel (Janela)
 * S_H:  Stride Vertical (Passo Y)
 * S_W:  Stride Horizontal (Passo X)
 * P_H:  Padding Vertical (Adiciona zeros em Cima e Embaixo)
 * P_W:  Padding Horizontal (Adiciona zeros na Esquerda e Direita)
 */
template <
    typename T,
    int IN_H, int IN_W,
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W
>
void sum_pooling_custom(
    const T input[IN_H][IN_W],
    // A dimensão de saída é calculada automaticamente pelo compilador C++
    T output[(IN_H + 2*P_H - K_H) / S_H + 1][(IN_W + 2*P_W - K_W) / S_W + 1]
) {
    // #pragma HLS INLINE // Opcional: inline para remover hierarquia
    
    // Constantes de dimensão de saída
    const int OUT_H = (IN_H + 2 * P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - K_W) / S_W + 1;

    // Loop Vertical da Saída
    row_loop: for (int i = 0; i < OUT_H; ++i) {
        
        // Loop Horizontal da Saída
        col_loop: for (int j = 0; j < OUT_W; ++j) {
            #pragma HLS PIPELINE II=1
            
            T sum = 0;

            // --- Janela do Kernel (Retangular) ---
            
            // Loop Vertical do Kernel
            k_row_loop: for (int ki = 0; ki < K_H; ++ki) {
                
                // Loop Horizontal do Kernel
                k_col_loop: for (int kj = 0; kj < K_W; ++kj) {
                    
                    // Lógica de Endereçamento com Padding Virtual
                    // Fórmula: (Posição Saída * Stride) + Offset Kernel - Padding
                    int r_idx = (i * S_H) + ki - P_H;
                    int c_idx = (j * S_W) + kj - P_W;

                    // Verificação de Borda (Boundary Check)
                    // Se estiver dentro da matriz, soma o valor.
                    // Se estiver fora (região de padding), soma 0 (ou seja, não faz nada).
                    if (r_idx >= 0 && r_idx < IN_H && c_idx >= 0 && c_idx < IN_W) {
                        sum += input[r_idx][c_idx];
                    }
                    // else { sum += 0; } // Implícito
                }
            }
            output[i][j] = sum;
        }
    }
}

/**
 * Conv2D Genérica para Vitis HLS
 * ------------------------------------------------------------------
 * Suporta: Weights, Bias, Stride, Padding, Dilation, Groups.
 * * T_DATA: Tipo dos dados (input/output)
 * T_WEIGHT: Tipo dos pesos e bias
 * * Dimensões (Template Parameters para Síntese Estática):
 * C_IN, H_IN, W_IN: Dimensões de Entrada
 * C_OUT: Canais de Saída
 * K_H, K_W: Tamanho do Kernel
 * S_H, S_W: Stride (Passo)
 * P_H, P_W: Padding (Zero padding)
 * D_H, D_W: Dilation (Espaçamento entre elementos do kernel)
 * GROUPS: Número de grupos (1 = Conv Normal, C_IN = Depthwise)
 */
template <
    typename T_DATA, typename T_WEIGHT,
    int C_IN, int H_IN, int W_IN,
    int C_OUT,
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int D_H, int D_W,
    int GROUPS
>
void conv2d_generic(
    const T_DATA input[C_IN][H_IN][W_IN],
    const T_WEIGHT weights[C_OUT][C_IN / GROUPS][K_H][K_W], // Pesos divididos por grupo
    const T_WEIGHT bias[C_OUT],
    // Dimensões de saída calculadas com Dilation
    T_DATA output[C_OUT][(H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1]
                        [(W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1]
) {
    // 1. Cálculo das Dimensões de Saída (Compile-Time)
    // Fórmula: Output = (Input + 2*Pad - Effective_Kernel) / Stride + 1
    // Effective_Kernel = Kernel + (Kernel-1)*(Dilation-1)
    const int K_H_EFF = K_H + (K_H - 1) * (D_H - 1);
    const int K_W_EFF = K_W + (K_W - 1) * (D_W - 1);
    
    const int H_OUT = (H_IN + 2 * P_H - K_H_EFF) / S_H + 1;
    const int W_OUT = (W_IN + 2 * P_W - K_W_EFF) / S_W + 1;

    // Constantes para controle de Grupos
    const int C_IN_GROUP = C_IN / GROUPS;   // Canais de entrada por grupo
    const int C_OUT_GROUP = C_OUT / GROUPS; // Canais de saída por grupo

    // ================= LOOP PRINCIPAL =================
    
    // Itera sobre Canais de Saída (Filters)
    loop_oc: for (int oc = 0; oc < C_OUT; ++oc) {
        
        // Identifica em qual grupo este canal de saída está
        int group_id = oc / C_OUT_GROUP;
        
        // Define o intervalo de canais de entrada correspondente a este grupo
        int in_ch_start = group_id * C_IN_GROUP;

        // Itera sobre Altura da Saída
        loop_oh: for (int oh = 0; oh < H_OUT; ++oh) {
            
            // Itera sobre Largura da Saída
            loop_ow: for (int ow = 0; ow < W_OUT; ++ow) {
                #pragma HLS PIPELINE II=1
                
                // Inicializa acumulador com o BIAS
                T_DATA sum = (T_DATA)bias[oc];

                // --- Operação de Convolução ---
                
                // Itera sobre canais de entrada DO GRUPO ATUAL
                loop_ic: for (int ic_offset = 0; ic_offset < C_IN_GROUP; ++ic_offset) {
                    
                    // Canal real de entrada
                    int ic = in_ch_start + ic_offset;

                    // Itera sobre Altura do Kernel
                    loop_kh: for (int kh = 0; kh < K_H; ++kh) {
                        
                        // Itera sobre Largura do Kernel
                        loop_kw: for (int kw = 0; kw < K_W; ++kw) {
                            
                            // Cálculo da posição com DILATION e PADDING
                            // Pos = (Saida * Stride) + (Kernel * Dilation) - Padding
                            int in_row = (oh * S_H) + (kh * D_H) - P_H;
                            int in_col = (ow * S_W) + (kw * D_W) - P_W;

                            // Verificação de Borda (Padding Virtual)
                            if (in_row >= 0 && in_row < H_IN && in_col >= 0 && in_col < W_IN) {
                                sum += input[ic][in_row][in_col] * weights[oc][ic_offset][kh][kw];
                            }
                        }
                    }
                }
                
                // Escrita na saída
                output[oc][oh][ow] = sum;
            }
        }
    }
}

// =========================================================
// KERNELS MATEMÁTICOS (Lógica de 1 Neurônio)
// =========================================================

/**
 * Integrator Kernel - Calcula a dinâmica de um único neurônio
 * v_new = v_old + (Input * R)
 */
template <typename T_DATA, typename T_PARAM>
void integrator_kernel(T_DATA input, T_PARAM R, T_DATA& v_state) {
    #pragma HLS INLINE
    v_state = v_state + (input * R);
}

/**
 * Integrate-and-Fire Kernel - Integrador com threshold
 */
template <typename T_DATA, typename T_PARAM>
void if_kernel(T_DATA input, T_PARAM R, T_PARAM threshold, T_DATA v_reset, 
               T_DATA& v_state, bool& spike) {
    #pragma HLS INLINE
    
    // 1. Integração
    v_state = v_state + (input * R);
    
    // 2. Disparo
    if (v_state >= threshold) {
        spike = true;
        v_state = v_reset;
    } else {
        spike = false;
    }
}

/**
 * Leaky Integrator Kernel - Dinâmica com vazamento
 * tau * dv/dt = (v_leak - v) + R*I
 */
template <typename T_DATA>
void li_kernel(T_DATA input, T_DATA tau, T_DATA R, T_DATA v_leak, T_DATA dt,
               T_DATA& v_state) {
    #pragma HLS INLINE
    
    // Discretização de Euler: v_new = v_old + (dt/tau) * ((v_leak - v_old) + R*Input)
    T_DATA leak_current = v_leak - v_state;
    T_DATA input_current = R * input;
    T_DATA dv = (dt / tau) * (leak_current + input_current);
    
    v_state = v_state + dv;
}

/**
 * LIF Kernel - Leaky Integrator com threshold
 */
template <typename T_DATA>
void lif_kernel(T_DATA input, T_DATA tau, T_DATA R, T_DATA v_leak, T_DATA dt,
                T_DATA v_threshold, T_DATA v_reset, T_DATA& v_state, bool& spike) {
    #pragma HLS INLINE
    
    // 1. Integração com vazamento
    T_DATA leak_current = v_leak - v_state;
    T_DATA input_current = R * input;
    T_DATA dv = (dt / tau) * (leak_current + input_current);
    
    v_state = v_state + dv;
    
    // 2. Disparo e Reset
    if (v_state >= v_threshold) {
        spike = true;
        v_state = v_reset;
    } else {
        spike = false;
    }
}

/**
 * CubaLI Kernel - Current-Based Leaky Integrator
 * Modelo de dois estágios: sinapse (u) + membrana (v)
 */
template <typename T_DATA>
void cuba_li_kernel(T_DATA input, T_DATA tau_syn, T_DATA w_in, T_DATA tau_mem, 
                    T_DATA R, T_DATA v_leak, T_DATA dt, T_DATA& u_state, T_DATA& v_state) {
    #pragma HLS INLINE
    
    // ESTÁGIO 1: Dinâmica da Sinapse (u)
    // tau_syn * du/dt = -u + w_in * input
    T_DATA leak_u = 0 - u_state;  // Sinapse decai para zero
    T_DATA input_u = w_in * input;
    T_DATA du = (dt / tau_syn) * (leak_u + input_u);
    u_state = u_state + du;
    
    // ESTÁGIO 2: Dinâmica da Membrana (v)
    // tau_mem * dv/dt = (v_leak - v) + R * u
    T_DATA leak_v = v_leak - v_state;
    T_DATA input_v = R * u_state;
    T_DATA dv = (dt / tau_mem) * (leak_v + input_v);
    v_state = v_state + dv;
}

/**
 * CubaLIF Kernel - Current-Based LIF
 * CubaLI + threshold e reset na membrana
 */
template <typename T_DATA, typename W_DATA>
void cuba_lif_kernel(T_DATA input, W_DATA tau_syn, W_DATA w_in, W_DATA tau_mem,
                     W_DATA R, W_DATA v_leak, W_DATA dt, W_DATA v_threshold, W_DATA v_reset,
                     W_DATA& u_state, W_DATA& v_state, bit_t& spike) {
    #pragma HLS INLINE
    
    // ESTÁGIO 1: Dinâmica da Sinapse (u)
    T_DATA leak_u = 0 - u_state;
    T_DATA input_u = w_in * input;
    T_DATA du = (dt / tau_syn) * (leak_u + input_u);
    u_state = u_state + du;
    
    // ESTÁGIO 2: Dinâmica da Membrana (v)
    T_DATA leak_v = v_leak - v_state;
    T_DATA input_v = R * u_state;
    T_DATA dv = (dt / tau_mem) * (leak_v + input_v);
    v_state = v_state + dv;
    
    // ESTÁGIO 3: Disparo e Reset (apenas na membrana)
    if (v_state >= v_threshold) {
        spike = 1;
        v_state = v_reset;
        // Nota: u_state NÃO é resetado em modelos CuBa
    } else {
        spike = 0;
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - INTEGRATE-AND-FIRE
// =========================================================

/**
 * Integrate-and-Fire 1D (Dense/FC Layers)
 */
template <typename T_DATA, typename T_PARAM, int N>
void integrate_and_fire(
    const T_DATA input[N],
    const T_PARAM R[N],
    const T_PARAM threshold[N],
    const T_DATA v_reset,
    T_DATA membrane_potential[N],
    bool output_spikes[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        if_kernel(input[i], R[i], threshold[i], v_reset, 
                  membrane_potential[i], output_spikes[i]);
    }
}

/**
 * Integrate-and-Fire 2D (Conv1D ou Imagens P&B)
 */
template <
    typename T_DATA, 
    typename T_PARAM,
    int H, int W
>
void integrate_and_fire(
    const T_DATA input[H][W],          // Entrada (Corrente)
    const T_PARAM R[H][W],             // Matriz R (Resistência/Ganho)
    const T_PARAM threshold[H][W],     // Matriz T (Limiares)
    const T_DATA v_reset,              // Valor de Reset (geralmente 0)
    T_DATA membrane_potential[H][W],   // ESTADO: Memória da voltagem (Read/Write)
    bool output_spikes[H][W]           // Saída: 1 (Spike) ou 0 (Silêncio)
) {
    // Diretivas de Interface para HLS (opcional, mas recomendado)
    // #pragma HLS ARRAY_PARTITION variable=R cyclic factor=4 dim=2
    
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            if_kernel(input[i][j], R[i][j], threshold[i][j], v_reset,
                      membrane_potential[i][j], output_spikes[i][j]);
        }
    }
}

/**
 * Integrate-and-Fire 3D (Conv2D com Canais)
 */
template <typename T_DATA, typename T_PARAM, int CH, int H, int W>
void integrate_and_fire(
    const T_DATA input[CH][H][W],
    const T_PARAM R[CH][H][W],
    const T_PARAM threshold[CH][H][W],
    const T_DATA v_reset,
    T_DATA membrane_potential[CH][H][W],
    bool output_spikes[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                if_kernel(input[c][i][j], R[c][i][j], threshold[c][i][j], v_reset,
                          membrane_potential[c][i][j], output_spikes[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - INTEGRATOR
// =========================================================

/**
 * Integrator 1D (Dense/FC Layers)
 */
template <typename T_DATA, typename T_PARAM, int N>
void integrator(
    const T_DATA input[N],
    const T_PARAM R[N],
    T_DATA voltage_state[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        integrator_kernel(input[i], R[i], voltage_state[i]);
    }
}

/**
 * Integrator 2D (Conv1D ou Imagens P&B)
 */
template <
    typename T_DATA,
    typename T_PARAM,
    int H, int W
>
void integrator(
    const T_DATA input[H][W],         // Corrente de Entrada (I)
    const T_PARAM R[H][W],            // Resistência (R) - Matriz por elemento
    T_DATA voltage_state[H][W]        // Estado da Voltagem (v) - Memória (In/Out)
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            integrator_kernel(input[i][j], R[i][j], voltage_state[i][j]);
        }
    }
}

/**
 * Integrator 3D (Conv2D com Canais)
 */
template <typename T_DATA, typename T_PARAM, int CH, int H, int W>
void integrator(
    const T_DATA input[CH][H][W],
    const T_PARAM R[CH][H][W],
    T_DATA voltage_state[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                integrator_kernel(input[c][i][j], R[c][i][j], voltage_state[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - LEAKY INTEGRATOR
// =========================================================

/**
 * Leaky Integrator 1D (Dense/FC Layers)
 */
template <typename T_DATA, int N>
void leaky_integrator(
    const T_DATA input[N],
    const T_DATA tau[N],
    const T_DATA R[N],
    const T_DATA v_leak[N],
    const T_DATA dt,
    T_DATA v_state[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        li_kernel(input[i], tau[i], R[i], v_leak[i], dt, v_state[i]);
    }
}

/**
 * Leaky Integrator 2D (Conv1D ou Imagens P&B)
 */
template <
    typename T_DATA,
    int H, int W
>
void leaky_integrator(
    const T_DATA input[H][W],         // Corrente de Entrada (I)
    const T_DATA tau[H][W],           // Constante de tempo (tau) [ms]
    const T_DATA R[H][W],             // Resistência (R) [Ohm]
    const T_DATA v_leak[H][W],        // Voltagem de Vazamento/Repouso (v_leak) [mV]
    const T_DATA dt,                  // Passo de tempo da simulação [ms]
    T_DATA v_state[H][W]              // Estado da Voltagem (v) - Memória In/Out
) {
    // Diretivas HLS para paralelismo
    // #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=4 dim=2
    
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            li_kernel(input[i][j], tau[i][j], R[i][j], v_leak[i][j], dt, v_state[i][j]);
        }
    }
}

/**
 * Leaky Integrator 3D (Conv2D com Canais)
 */
template <typename T_DATA, int CH, int H, int W>
void leaky_integrator(
    const T_DATA input[CH][H][W],
    const T_DATA tau[CH][H][W],
    const T_DATA R[CH][H][W],
    const T_DATA v_leak[CH][H][W],
    const T_DATA dt,
    T_DATA v_state[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                li_kernel(input[c][i][j], tau[c][i][j], R[c][i][j], 
                          v_leak[c][i][j], dt, v_state[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - LIF (Leaky Integrate-and-Fire)
// =========================================================

/**
 * LIF 1D (Dense/FC Layers)
 */
template <typename T_DATA, int N>
void lif_neuron(
    const T_DATA input[N],
    const T_DATA tau[N],
    const T_DATA R[N],
    const T_DATA v_leak[N],
    const T_DATA dt,
    const T_DATA v_threshold[N],
    const T_DATA v_reset,
    T_DATA v_state[N],
    bool spikes_out[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        lif_kernel(input[i], tau[i], R[i], v_leak[i], dt,
                   v_threshold[i], v_reset, v_state[i], spikes_out[i]);
    }
}

/**
 * LIF 2D (Conv1D ou Imagens P&B)
 */
template <
    typename T_DATA,
    int H, int W
>
void lif_neuron(
    // Entradas da Dinâmica (passadas para o LI)
    const T_DATA input[H][W],
    const T_DATA tau[H][W],
    const T_DATA R[H][W],
    const T_DATA v_leak[H][W],
    const T_DATA dt,
    
    // Parâmetros de Disparo (Novos)
    const T_DATA v_threshold[H][W],  // Limiar de disparo
    const T_DATA v_reset,            // Valor para onde a voltagem vai após spike
    
    // Estados (IO)
    T_DATA v_state[H][W],            // Memória da Voltagem
    bool spikes_out[H][W]            // Saída de Spikes (1 ou 0)
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            lif_kernel(input[i][j], tau[i][j], R[i][j], v_leak[i][j], dt,
                       v_threshold[i][j], v_reset, v_state[i][j], spikes_out[i][j]);
        }
    }
}

/**
 * LIF 3D (Conv2D com Canais)
 */
template <typename T_DATA, int CH, int H, int W>
void lif_neuron(
    const T_DATA input[CH][H][W],
    const T_DATA tau[CH][H][W],
    const T_DATA R[CH][H][W],
    const T_DATA v_leak[CH][H][W],
    const T_DATA dt,
    const T_DATA v_threshold[CH][H][W],
    const T_DATA v_reset,
    T_DATA v_state[CH][H][W],
    bool spikes_out[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                lif_kernel(input[c][i][j], tau[c][i][j], R[c][i][j], 
                           v_leak[c][i][j], dt, v_threshold[c][i][j], v_reset,
                           v_state[c][i][j], spikes_out[c][i][j]);
            }
        }
    }
}

/**
 * Linear Scale (Helper Function)
 * ----------------------------------------------------------------
 * Aplica escala elemento-por-elemento: output[i] = input[i] * weights[i]
 * Usada como estágio inicial em modelos Current-Based.
 */
template <typename T_DATA, int DIM>
void linear_scale(const T_DATA input[DIM], const T_DATA weights[DIM], T_DATA output[DIM]) {
    #pragma HLS PIPELINE II=1
    scale_loop: for(int i=0; i<DIM; i++) {
        output[i] = input[i] * weights[i];
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - CUBA-LI
// =========================================================

/**
 * CubaLI 1D (Dense/FC Layers)
 */
template <typename T_DATA, int N>
void cuba_li_neuron(
    const T_DATA input[N],
    const T_DATA tau_syn[N],
    const T_DATA w_in[N],
    const T_DATA tau_mem[N],
    const T_DATA R[N],
    const T_DATA v_leak[N],
    const T_DATA dt,
    T_DATA u_state[N],
    T_DATA v_state[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        cuba_li_kernel(input[i], tau_syn[i], w_in[i], tau_mem[i], 
                       R[i], v_leak[i], dt, u_state[i], v_state[i]);
    }
}

/**
 * CubaLI 2D (Conv1D ou Imagens P&B)
 */
template <
    typename T_DATA,
    int H, int W
>
void cuba_li_neuron(
    const T_DATA input[H][W],
    const T_DATA tau_syn[H][W],
    const T_DATA w_in[H][W],
    const T_DATA tau_mem[H][W],
    const T_DATA R[H][W],
    const T_DATA v_leak[H][W],
    const T_DATA dt,
    T_DATA u_state[H][W],
    T_DATA v_state[H][W]
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            cuba_li_kernel(input[i][j], tau_syn[i][j], w_in[i][j], tau_mem[i][j],
                           R[i][j], v_leak[i][j], dt, u_state[i][j], v_state[i][j]);
        }
    }
}

/**
 * CubaLI 3D (Conv2D com Canais)
 */
template <typename T_DATA, int CH, int H, int W>
void cuba_li_neuron(
    const T_DATA input[CH][H][W],
    const T_DATA tau_syn[CH][H][W],
    const T_DATA w_in[CH][H][W],
    const T_DATA tau_mem[CH][H][W],
    const T_DATA R[CH][H][W],
    const T_DATA v_leak[CH][H][W],
    const T_DATA dt,
    T_DATA u_state[CH][H][W],
    T_DATA v_state[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                cuba_li_kernel(input[c][i][j], tau_syn[c][i][j], w_in[c][i][j], 
                               tau_mem[c][i][j], R[c][i][j], v_leak[c][i][j], dt,
                               u_state[c][i][j], v_state[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - CUBA-LIF
// =========================================================

/**
 * CubaLIF 1D (Dense/FC Layers)
 */
template <typename T_DATA, typename W_DATA, int N>
void CubaLIF(
    const T_DATA (&input)[N],
    bit_t spikes_out[N],

    const W_DATA tau_syn[N],
    const W_DATA tau_mem[N],
    const W_DATA R_mem[N],
    const W_DATA v_leak[N],
    const W_DATA v_threshold[N],
    const W_DATA v_reset[N],
    const W_DATA w_in[N],

    W_DATA u_state[N],
    W_DATA v_state[N],
    const W_DATA dt
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        cuba_lif_kernel(input[i], tau_syn[i], w_in[i], tau_mem[i], R_mem[i],
                        v_leak[i], dt, v_threshold[i], v_reset[i],
                        u_state[i], v_state[i], spikes_out[i]);
    }
}

// /**
//  * CubaLIF 2D (Conv1D ou Imagens P&B)
//  */
// template <
//     typename T_DATA,
//     int H, int W
// >
// void CubaLIF(
//     const T_DATA input[H][W],
//     const T_DATA w_in[H][W],
//     const T_DATA tau_syn[H][W],
//     const T_DATA tau_mem[H][W],
//     const T_DATA R_mem[H][W],
//     const T_DATA v_leak[H][W],
//     const T_DATA v_threshold[H][W],
//     const T_DATA v_reset,
//     const T_DATA dt,
//     T_DATA u_state[H][W],
//     T_DATA v_state[H][W],
//     bool spikes_out[H][W]
// ) {
//     loop_h: for(int i = 0; i < H; i++) {
//         loop_w: for(int j = 0; j < W; j++) {
//             #pragma HLS PIPELINE II=1
//             cuba_lif_kernel(input[i][j], tau_syn[i][j], w_in[i][j], tau_mem[i][j],
//                             R_mem[i][j], v_leak[i][j], dt, v_threshold[i][j], v_reset,
//                             u_state[i][j], v_state[i][j], spikes_out[i][j]);
//         }
//     }
// }

// /**
//  * CubaLIF 3D (Conv2D com Canais)
//  */
// template <typename T_DATA, int CH, int H, int W>
// void CubaLIF(
//     const T_DATA input[CH][H][W],
//     const T_DATA w_in[CH][H][W],
//     const T_DATA tau_syn[CH][H][W],
//     const T_DATA tau_mem[CH][H][W],
//     const T_DATA R_mem[CH][H][W],
//     const T_DATA v_leak[CH][H][W],
//     const T_DATA v_threshold[CH][H][W],
//     const T_DATA v_reset,
//     const T_DATA dt,
//     T_DATA u_state[CH][H][W],
//     T_DATA v_state[CH][H][W],
//     bool spikes_out[CH][H][W]
// ) {
//     loop_ch: for(int c = 0; c < CH; c++) {
//         loop_h: for(int i = 0; i < H; i++) {
//             loop_w: for(int j = 0; j < W; j++) {
//                 #pragma HLS PIPELINE II=1
//                 cuba_lif_kernel(input[c][i][j], tau_syn[c][i][j], w_in[c][i][j],
//                                 tau_mem[c][i][j], R_mem[c][i][j], v_leak[c][i][j], dt,
//                                 v_threshold[c][i][j], v_reset, u_state[c][i][j],
//                                 v_state[c][i][j], spikes_out[c][i][j]);
//             }
//         }
//     }
// }
