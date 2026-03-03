#include <iostream>

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
template <typename T_DATA>
void cuba_lif_kernel(T_DATA input, T_DATA tau_syn, T_DATA w_in, T_DATA tau_mem,
                     T_DATA R, T_DATA v_leak, T_DATA dt, T_DATA v_threshold, T_DATA v_reset,
                     T_DATA& u_state, T_DATA& v_state, bool& spike) {
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
        spike = true;
        v_state = v_reset;
        // Nota: u_state NÃO é resetado em modelos CuBa
    } else {
        spike = false;
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

/**
 * Affine Layer (Fully Connected) para Vitis HLS
 * ----------------------------------------------------------------
 * Operação: Output = (Weights * Input) + Bias
 * * T_DATA: Tipo de dado (float, int, ap_fixed)
 * IN_DIM: Tamanho do vetor de entrada (Input Features)
 * OUT_DIM: Tamanho do vetor de saída (Output Neurons)
 */
template <
    typename T_DATA,
    int IN_DIM,
    int OUT_DIM
>
void Affine(
    const T_DATA input[IN_DIM],              // Vetor I
    const T_DATA weights[OUT_DIM][IN_DIM],   // Matriz W [Linhas=Saída][Cols=Entrada]
    const T_DATA bias[OUT_DIM],              // Vetor b
    T_DATA output[OUT_DIM]                   // Resultado
) {
    // Diretiva para particionar o array de entrada se quiser paralelismo total
    // #pragma HLS ARRAY_PARTITION variable=input complete
    
    // Loop sobre os neurônios de SAÍDA
    loop_out: for (int i = 0; i < OUT_DIM; ++i) {
        
        // O Pipeline aqui permite calcular um neurônio de saída a cada ciclo (se unrolled)
        // ou otimizar o loop interno de soma.
        #pragma HLS PIPELINE II=1
        
        // 1. Inicia com o valor do Bias (b)
        T_DATA acc = bias[i];

        // 2. Produto Escalar (Dot Product): W * I
        loop_in: for (int j = 0; j < IN_DIM; ++j) {
            // Em HLS, para latência mínima, pode-se usar #pragma HLS UNROLL aqui
            acc += input[j] * weights[i][j];
        }

        // 3. Atribui ao output
        output[i] = acc;
    }
}

/**
 * Linear Layer para Vitis HLS
 * ----------------------------------------------------------------
 * Operação: Output = Weights * Input
 * Diferença para Affine: Não possui soma de Bias.
 * * T_DATA: Tipo de dado (float, int, ap_fixed)
 * IN_DIM: Tamanho do vetor de entrada (Features)
 * OUT_DIM: Tamanho do vetor de saída (Neurons)
 */
template <
    typename T_DATA,
    int IN_DIM,
    int OUT_DIM
>
void linear_layer(
    const T_DATA input[IN_DIM],              // Vetor I
    const T_DATA weights[OUT_DIM][IN_DIM],   // Matriz W
    T_DATA output[OUT_DIM]                   // Vetor Saída
) {
    // Diretivas HLS opcionais para particionamento de memória
    // #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=IN_DIM dim=2
    
    // Loop sobre cada neurônio de saída
    loop_out: for (int i = 0; i < OUT_DIM; ++i) {
        
        // Pipeline permite iniciar o cálculo do próximo neurônio enquanto o atual finaliza
        #pragma HLS PIPELINE II=1
        
        T_DATA acc = 0; // Acumulador inicia em ZERO (diferente do Affine que inicia com Bias)

        // Produto Escalar (Dot Product): Linha da Matriz * Vetor Entrada
        loop_in: for (int j = 0; j < IN_DIM; ++j) {
            // Unroll aqui paralelizaria as multiplicações se houver DSPs suficientes
            acc += input[j] * weights[i][j];
        }

        output[i] = acc;
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
template <typename T_DATA, int N>
void cuba_lif_neuron(
    const T_DATA input[N],
    const T_DATA w_in[N],
    const T_DATA tau_syn[N],
    const T_DATA tau_mem[N],
    const T_DATA R_mem[N],
    const T_DATA v_leak[N],
    const T_DATA v_threshold[N],
    const T_DATA v_reset,
    const T_DATA dt,
    T_DATA u_state[N],
    T_DATA v_state[N],
    bool spikes_out[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        cuba_lif_kernel(input[i], tau_syn[i], w_in[i], tau_mem[i], R_mem[i],
                        v_leak[i], dt, v_threshold[i], v_reset,
                        u_state[i], v_state[i], spikes_out[i]);
    }
}

/**
 * CubaLIF 2D (Conv1D ou Imagens P&B)
 */
template <
    typename T_DATA,
    int H, int W
>
void cuba_lif_neuron(
    const T_DATA input[H][W],
    const T_DATA w_in[H][W],
    const T_DATA tau_syn[H][W],
    const T_DATA tau_mem[H][W],
    const T_DATA R_mem[H][W],
    const T_DATA v_leak[H][W],
    const T_DATA v_threshold[H][W],
    const T_DATA v_reset,
    const T_DATA dt,
    T_DATA u_state[H][W],
    T_DATA v_state[H][W],
    bool spikes_out[H][W]
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            cuba_lif_kernel(input[i][j], tau_syn[i][j], w_in[i][j], tau_mem[i][j],
                            R_mem[i][j], v_leak[i][j], dt, v_threshold[i][j], v_reset,
                            u_state[i][j], v_state[i][j], spikes_out[i][j]);
        }
    }
}

/**
 * CubaLIF 3D (Conv2D com Canais)
 */
template <typename T_DATA, int CH, int H, int W>
void cuba_lif_neuron(
    const T_DATA input[CH][H][W],
    const T_DATA w_in[CH][H][W],
    const T_DATA tau_syn[CH][H][W],
    const T_DATA tau_mem[CH][H][W],
    const T_DATA R_mem[CH][H][W],
    const T_DATA v_leak[CH][H][W],
    const T_DATA v_threshold[CH][H][W],
    const T_DATA v_reset,
    const T_DATA dt,
    T_DATA u_state[CH][H][W],
    T_DATA v_state[CH][H][W],
    bool spikes_out[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                cuba_lif_kernel(input[c][i][j], tau_syn[c][i][j], w_in[c][i][j],
                                tau_mem[c][i][j], R_mem[c][i][j], v_leak[c][i][j], dt,
                                v_threshold[c][i][j], v_reset, u_state[c][i][j],
                                v_state[c][i][j], spikes_out[c][i][j]);
            }
        }
    }
}

/**
 * AveragePooling 2D
 * ----------------------------------------------------------------
 * Composição: 
 * 1. SumPooling (Agrupa e soma os elementos da janela)
 * 2. Linear Scale (Multiplica a soma pelo inverso da área do kernel)
 */
template <
    typename T,
    int IN_H, int IN_W,
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W
>
void avg_pooling_custom(
    const T input[IN_H][IN_W],
    T output[(IN_H + 2*P_H - K_H) / S_H + 1][(IN_W + 2*P_W - K_W) / S_W + 1]
) {
    #pragma HLS INLINE // Fused modules: otimiza o hardware fundindo as operações

    // Constantes de dimensão
    const int OUT_H = (IN_H + 2 * P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - K_W) / S_W + 1;
    const int OUT_DIM = OUT_H * OUT_W;

    // Buffer intermediário para armazenar a saída do SumPooling
    T sum_buffer[OUT_H][OUT_W];
    
    // O fator de escala da média é 1 / (Altura * Largura do Kernel)
    // Feito em tempo de compilação para evitar divisões no FPGA
    T scale_factor = (T)1.0 / (T)(K_H * K_W);
    
    // A função linear_scale exige um array de "pesos". 
    // Criamos um array preenchido com a constante de escala.
    T scale_weights[OUT_DIM];
    
    init_weights: for (int i = 0; i < OUT_DIM; ++i) {
        #pragma HLS UNROLL // Desenrola pois é uma inicialização simples e estática
        scale_weights[i] = scale_factor;
    }

    // ---------------------------------------------------------
    // ESTÁGIO 1: Soma da Janela
    // ---------------------------------------------------------
    sum_pooling_custom<T, IN_H, IN_W, K_H, K_W, S_H, S_W, P_H, P_W>(input, sum_buffer);

    // ---------------------------------------------------------
    // ESTÁGIO 2: Escala (Média)
    // ---------------------------------------------------------
    // Tratamos a matriz 2D como um vetor 1D para reaproveitar a linear_scale
    linear_scale<T, OUT_DIM>((const T*)sum_buffer, scale_weights, (T*)output);
}

// --------------------------------------------------------
// Testbench - SumPooling Paramétrico
// --------------------------------------------------------
void sum_pooling_test()
{
    // Exemplo: Imagem 5x5
    const int H = 5;
    const int W = 5;
    
    // Configuração "Exótica" para provar a flexibilidade:
    // Kernel Retangular: 2x3 (Altura 2, Largura 3)
    const int K_H = 2;
    const int K_W = 3;
    
    // Stride Assimétrico: Anda 2 linhas, mas apenas 1 coluna
    const int S_H = 2;
    const int S_W = 1;
    
    // Padding: 1 pixel nas bordas verticais, 0 nas horizontais
    const int P_H = 1;
    const int P_W = 0;

    // Cálculo das dimensões de saída esperadas
    const int OUT_H = (H + 2 * P_H - K_H) / S_H + 1; // (5 + 2 - 2)/2 + 1 = 3
    const int OUT_W = (W + 2 * P_W - K_W) / S_W + 1; // (5 + 0 - 3)/1 + 1 = 3

    // Dados de entrada (Matriz 5x5 preenchida com 1)
    int img_in[H][W];
    for(int i=0; i<H; i++)
        for(int j=0; j<W; j++)
            img_in[i][j] = 1;

    // Matriz de saída
    int img_out[OUT_H][OUT_W];

    // Chamada
    sum_pooling_custom<int, H, W, K_H, K_W, S_H, S_W, P_H, P_W>(img_in, img_out);

    // Impressão
    std::cout << "=== Teste SumPooling ===\n";
    std::cout << "Input 5x5 (Tudo 1). Kernel 2x3. Padding Vert=1.\n";
    std::cout << "Saida (" << OUT_H << "x" << OUT_W << "):\n";
    
    for(int i=0; i<OUT_H; i++) {
        std::cout << "[ ";
        for(int j=0; j<OUT_W; j++) {
            std::cout << img_out[i][j] << " ";
        }
        std::cout << "]\n";
    }
}

// --------------------------------------------------------
// Testbench - AveragePooling (Sum + Linear Scale)
// --------------------------------------------------------
void avg_pooling_test()
{
    // Configuração: Imagem 4x4, Kernel 2x2, Stride 2, sem padding.
    const int IN_H = 4, IN_W = 4;
    const int K_H = 2,  K_W = 2;
    const int S_H = 2,  S_W = 2;
    const int P_H = 0,  P_W = 0;

    const int OUT_H = (IN_H + 2*P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2*P_W - K_W) / S_W + 1;

    // Imagem 4x4
    float input[IN_H][IN_W] = {
        {2.0, 2.0,  6.0, 6.0},
        {2.0, 2.0,  6.0, 6.0},
        {4.0, 4.0,  8.0, 8.0},
        {4.0, 4.0,  8.0, 8.0}
    };

    float output[OUT_H][OUT_W];

    // Execução
    avg_pooling_custom<float, IN_H, IN_W, K_H, K_W, S_H, S_W, P_H, P_W>(input, output);

    // Verificação visual
    std::cout << "\n=== Teste AveragePooling (Sum + Linear Scale) ===\n";
    std::cout << "Kernel Area = " << (K_H * K_W) << " -> Fator Multiplicador: " << 1.0/(K_H*K_W) << "\n";
    std::cout << "Saida (" << OUT_H << "x" << OUT_W << "):\n";

    // Esperado:
    // Q1: (2+2+2+2)/4 = 2
    // Q2: (6+6+6+6)/4 = 6
    // Q3: (4+4+4+4)/4 = 4
    // Q4: (8+8+8+8)/4 = 8

    for (int i = 0; i < OUT_H; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < OUT_W; ++j) {
            std::cout << output[i][j] << " ";
        }
        std::cout << "]\n";
    }
}

// --------------------------------------------------------
// Testbench - Conv2D com Dilation
// --------------------------------------------------------
void conv2d_test()
{
    // Configuração do Teste:
    // 1 Grupo, 1 Canal In, 1 Canal Out.
    // Kernel 2x2.
    // Dilation = 2 (Buracos no kernel).
    // Padding = 0.
    
    const int C_I = 1; 
    const int H_I = 5; 
    const int W_I = 5;
    const int C_O = 1;
    const int K = 2;
    const int S = 1;
    const int P = 0;
    const int D = 2; // Dilation importante aqui!
    const int G = 1;

    // Dados
    int img[C_I][H_I][W_I];
    int w[C_O][C_I/G][K][K];
    int b[C_O] = {0}; // Sem bias para facilitar visualização
    
    // Output dimensions
    // Eff Kernel = 2 + (2-1)*(2-1) = 3 (Kernel vira 3x3 virtualmente: Valor, Buraco, Valor)
    // Out Size = (5 - 3) / 1 + 1 = 3
    const int H_O = 3; 
    const int W_O = 3;
    int out[C_O][H_O][W_O];

    // Preenchendo imagem com 1
    for(int i=0; i<H_I; i++)
        for(int j=0; j<W_I; j++)
            img[0][i][j] = 1;

    // Preenchendo Pesos com 1
    for(int i=0; i<K; i++)
        for(int j=0; j<K; j++)
            w[0][0][i][j] = 1;

    // Executando
    conv2d_generic<int, int, C_I, H_I, W_I, C_O, K, K, S, S, P, P, D, D, G>
                  (img, w, b, out);

    // Verificação visual
    std::cout << "\n=== Teste Conv2D ===\n";
    std::cout << "Dilation=2 (Kernel 2x2 vira 3x3 efetivo)\n";
    std::cout << "Imagem 5x5 (1s). Soma esperada = 4 (pois o kernel tem 4 elementos de valor 1)\n";
    std::cout << "Saida (" << H_O << "x" << W_O << "):\n";
    
    for(int i=0; i<H_O; i++) {
        std::cout << "[ ";
        for(int j=0; j<W_O; j++) {
            std::cout << out[0][i][j] << " ";
        }
        std::cout << "]\n";
    }
}

// --------------------------------------------------------
// Testbench - Integrate-and-Fire (Simulação Temporal)
// --------------------------------------------------------
void integrate_and_fire_test()
{
    // Dimensões 2x2
    const int H = 2;
    const int W = 2;
    
    // Parâmetros
    float input_frame[H][W] = {{1.0, 0.5}, {2.0, 0.1}}; // Corrente chegando
    float R[H][W]           = {{1.0, 2.0}, {1.0, 10.0}}; // Resistência
    float T[H][W]           = {{1.5, 1.5}, {1.5, 1.5}};  // Threshold constante
    float v_reset = 0.0;
    
    // ESTADO INICIAL (Memória dos neurônios começa zerada)
    // Em HLS, isso geralmente fica em uma BRAM estática ou é passado entre frames
    float membrane[H][W] = {{0.0, 0.0}, {0.0, 0.0}};
    bool spikes[H][W];

    std::cout << "\n=== Teste Integrate-and-Fire ===\n";
    std::cout << "Simulação de 2 Passos de Tempo\n";

    // Simular Passo 1
    std::cout << "\nTime Step 1:\n";
    integrate_and_fire<float, float, H, W>(input_frame, R, T, v_reset, membrane, spikes);
    
    // Exibe resultados passo 1
    // Neurônio (0,0): In=1, R=1 -> V=1.0. T=1.5. (Sem Spike)
    // Neurônio (0,1): In=0.5, R=2 -> V=1.0. T=1.5. (Sem Spike)
    // Neurônio (1,0): In=2, R=1 -> V=2.0. T=1.5. (SPIKE!) -> V reseta pra 0
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            std::cout << "Neuron [" << i << "][" << j << "]: " 
                      << "V_mem=" << membrane[i][j] 
                      << " Spike=" << spikes[i][j] << "\n";
        }
    }

    // Simular Passo 2 (Mesma entrada para ver acumulação)
    std::cout << "\nTime Step 2 (Acumulando):\n";
    integrate_and_fire<float, float, H, W>(input_frame, R, T, v_reset, membrane, spikes);
    
    // Exibe resultados passo 2
    // Neurônio (0,0): V_ant=1.0. In=1*1. Novo V=2.0. T=1.5 -> SPIKE!
    // Neurônio (1,0): V_ant=0 (resetado antes). In=2*1. Novo V=2.0 -> SPIKE!
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            std::cout << "Neuron [" << i << "][" << j << "]: " 
                      << "V_mem=" << membrane[i][j] 
                      << " Spike=" << spikes[i][j] << "\n";
        }
    }
}

// --------------------------------------------------------
// Testbench - Affine (Fully Connected)
// --------------------------------------------------------
void affine_test()
{
    // Configuração:
    // Entrada: vetor de 3 elementos [1, 2, 3]
    // Saída: vetor de 2 neurônios
    const int N_IN = 3;
    const int N_OUT = 2;

    // Dados
    int I[N_IN] = {1, 2, 3};
    
    // Pesos (W): Matriz 2x3
    // Neurônio 0 quer: 1*I[0] + 1*I[1] + 1*I[2]
    // Neurônio 1 quer: 2*I[0] + 0*I[1] + -1*I[2]
    int W[N_OUT][N_IN] = {
        {1, 1, 1},  
        {2, 0, -1}
    };

    // Bias (b):
    // Neurônio 0 soma +10
    // Neurônio 1 soma +5
    int b[N_OUT] = {10, 5};

    int out[N_OUT];

    // Execução
    Affine<int, N_IN, N_OUT>(I, W, b, out);

    // Verificação e Prova Real
    std::cout << "\n=== Teste Affine (Fully Connected) ===\n";
    
    // Neurônio 0:
    // (1*1 + 2*1 + 3*1) + 10 
    // (1 + 2 + 3) + 10 = 6 + 10 = 16
    std::cout << "Neuronio 0 Esperado: 16. Obtido: " << out[0] << "\n";

    // Neurônio 1:
    // (1*2 + 2*0 + 3*-1) + 5
    // (2 + 0 - 3) + 5 = -1 + 5 = 4
    std::cout << "Neuronio 1 Esperado:  4. Obtido: " << out[1] << "\n";
}

// --------------------------------------------------------
// Testbench - Linear (W * I, sem Bias)
// --------------------------------------------------------
void linear_test()
{
    // Configuração: 3 Entradas -> 2 Saídas
    const int N_IN = 3;
    const int N_OUT = 2;

    // Entrada I = [1, 2, 3]
    int I[N_IN] = {1, 2, 3};
    
    // Matriz de Pesos W (2x3)
    // Linha 0: [1, 2, 3] -> Esperado: 1*1 + 2*2 + 3*3 = 14
    // Linha 1: [4, 5, 6] -> Esperado: 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
    int W[N_OUT][N_IN] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    int out[N_OUT];

    // Execução
    linear_layer<int, N_IN, N_OUT>(I, W, out);

    // Verificação
    std::cout << "\n=== Teste Linear (W * I, sem Bias) ===\n";
    std::cout << "Input: [1, 2, 3]\n";
    
    std::cout << "Neuronio 0 (Pesos 1,2,3): " << out[0] 
              << " (Esperado: 14)\n";
              
    std::cout << "Neuronio 1 (Pesos 4,5,6): " << out[1] 
              << " (Esperado: 32)\n";
}

// --------------------------------------------------------
// Testbench - Integrator
// --------------------------------------------------------
void integrator_test()
{
    // Dimensões 2x2
    const int H = 2;
    const int W = 2;

    // Dados de Entrada (Corrente I)
    float I[H][W] = {
        {1.0f, 0.5f}, 
        {2.0f, 0.0f}
    };
    
    // Parâmetro R (Resistência)
    float R[H][W] = {
        {1.0f, 2.0f}, 
        {0.5f, 1.0f}
    };

    // Estado Inicial da Voltagem (v)
    // Em hardware, isso seria armazenado em BRAM ou registros
    float v[H][W] = {
        {0.0f, 0.0f}, 
        {10.0f, 5.0f} // Já começam com alguma carga
    };

    std::cout << "\n=== Teste Integrator ===\n";

    // --- Passo 1 ---
    std::cout << "\nTime Step 1:\n";
    integrator<float, float, H, W>(I, R, v);
    
    // Verificação Passo 1
    // (0,0): v_old=0.  Inc=1.0*1.0=1.  v_new=1.0
    // (0,1): v_old=0.  Inc=0.5*2.0=1.  v_new=1.0
    // (1,0): v_old=10. Inc=2.0*0.5=1.  v_new=11.0
    // (1,1): v_old=5.  Inc=0.0*1.0=0.  v_new=5.0
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            std::cout << "V[" << i << "][" << j << "] = " << v[i][j] << "\n";
        }
    }

    // --- Passo 2 (Acumulação) ---
    // Vamos manter a mesma entrada para ver a voltagem subir
    std::cout << "\nTime Step 2:\n";
    integrator<float, float, H, W>(I, R, v);

    // Verificação Passo 2
    // (0,0): v_old=1.0. Inc=1.  v_new=2.0
    // (1,0): v_old=11.0. Inc=1. v_new=12.0
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            std::cout << "V[" << i << "][" << j << "] = " << v[i][j] << "\n";
        }
    }
}

// --------------------------------------------------------
// Testbench - Leaky Integrator
// --------------------------------------------------------
void leaky_integrator_test()
{
    const int H = 1;
    const int W = 1;
    
    // Parâmetros do Neurônio
    float tau[H][W]    = {{ 10.0f }}; // tau = 10 ms (Decaimento lento)
    float R[H][W]      = {{ 1.0f }};  // R = 1 Ohm
    float v_leak[H][W] = {{ 0.0f }};  // Tende a voltar para 0 mV
    
    // Configuração da Simulação
    float dt = 1.0f; // Passo de 1 ms
    
    // Estado Inicial: Neurônio começa carregado em 10.0 mV
    float v_state[H][W] = {{ 10.0f }};
    
    // Entrada: ZERO corrente (I=0). Queremos ver apenas o vazamento (Leak).
    float input[H][W] = {{ 0.0f }};

    std::cout << "\n=== Teste Leaky Integrator ===\n";
    std::cout << "\nRelaxamento (sem entrada):\n";
    std::cout << "Inicio V = " << v_state[0][0] << " mV. Target = " << v_leak[0][0] << " mV.\n";
    
    // Simular 5 passos de tempo
    // Esperado: A voltagem deve cair exponencialmente em direção a 0.
    for(int step = 1; step <= 5; step++) {
        leaky_integrator<float, H, W>(input, tau, R, v_leak, dt, v_state);
        
        std::cout << "Step " << step << " (t=" << step*dt << "ms): V = " 
                  << v_state[0][0] << " mV\n";
    }
    
    // Prova Real Matemática (Step 1):
    // Eq: v_new = v_old + (dt/tau) * (v_leak - v_old)  (pois I=0)
    // v_new = 10 + (1/10) * (0 - 10)
    // v_new = 10 + 0.1 * (-10)
    // v_new = 10 - 1 = 9.0
    
    std::cout << "\nIntegração (com entrada):\n";
    // Agora injetamos corrente para ver ele subir contra o vazamento
    input[0][0] = 5.0f; // Injeta 5 de corrente
    // V_target teórico = v_leak + R*I = 0 + 1*5 = 5.0 mV
    
    // Resetamos o estado para 0
    v_state[0][0] = 0.0f;
    
    for(int step = 1; step <= 5; step++) {
        leaky_integrator<float, H, W>(input, tau, R, v_leak, dt, v_state);
        std::cout << "Step " << step << " (t=" << step*dt << "ms): V = " << v_state[0][0] << " mV\n";
    }
}

// --------------------------------------------------------
// Testbench - LIF (Leaky Integrate-and-Fire)
// --------------------------------------------------------
void lif_test()
{
    const int H = 1;
    const int W = 1;
    
    // Configuração
    float tau[H][W]    = {{ 10.0f }};
    float R[H][W]      = {{ 1.0f }};
    float v_leak[H][W] = {{ 0.0f }};
    float v_thresh[H][W] = {{ 2.0f }}; // Limiar baixo para disparar rápido
    float v_reset = 0.0f;
    float dt = 1.0f;
    
    // Estado Inicial
    float v_state[H][W] = {{ 0.0f }};
    bool spikes[H][W] = {{ false }};
    
    // Entrada Forte (para forçar disparo contra o vazamento)
    float input[H][W] = {{ 5.0f }}; 

    std::cout << "\n=== Teste LIF (Leaky Integrate-and-Fire) ===\n";
    std::cout << "Threshold = 2.0mV, V_reset = 0.0mV\n\n";

    // Simular 6 passos
    for(int step = 1; step <= 6; step++) {
        lif_neuron<float, H, W>(input, tau, R, v_leak, dt, v_thresh, v_reset, v_state, spikes);
        
        std::cout << "Step " << step << " (t=" << step*dt << "ms): V_mem = " << v_state[0][0] 
                  << " | Spike = " << spikes[0][0] << "\n";
    }
}

// --------------------------------------------------------
// Testbench - CuBa-LI (Current-Based Leaky Integrator)
// --------------------------------------------------------
void cuba_li_test()
{
    const int H = 1;
    const int W = 1;
    float dt = 1.0f;

    // Configuração
    // Sinapse Rápida (tau=2ms), Membrana Lenta (tau=5ms)
    float tau_syn[H][W] = {{ 2.0f }}; 
    float tau_mem[H][W] = {{ 5.0f }};
    float w_in[H][W]    = {{ 1.0f }}; // Ganho de entrada unitário
    float R[H][W]       = {{ 1.0f }}; // Resistência unitária
    float v_leak[H][W]  = {{ 0.0f }};

    // Estados Iniciais
    float u[H][W] = {{ 0.0f }};
    float v[H][W] = {{ 0.0f }};
    
    // Entrada: Pulso único de corrente no Step 1
    float input[H][W] = {{ 10.0f }};

    std::cout << "\n=== Teste CuBa-LI (Current-Based Leaky Integrator) ===\n";
    std::cout << "Second Order Dynamics - Sinapse + Membrana\n";
    std::cout << "Input 10.0 apenas no Step 1. Observe o atraso na subida de V.\n\n";

    for(int t=1; t<=5; t++) {
        cuba_li_neuron<float, H, W>(input, tau_syn, w_in, tau_mem, R, v_leak, dt, u, v);
        
        std::cout << "T=" << t << "ms | Synapse(u): " << u[0][0] 
                  << " | Membrane(v): " << v[0][0] << "\n";
        
        // Remove a entrada após o primeiro passo para ver o decaimento
        if(t == 1) input[0][0] = 0.0f;
    }
}

// --------------------------------------------------------
// Testbench - CuBa-LIF (Current-Based LIF) - 2D Original
// --------------------------------------------------------
void cuba_lif_test()
{
    const int H = 1;
    const int W = 1;
    float dt = 1.0f;

    // Configuração
    float w_in[H][W]    = {{ 2.0f }};  // Ganho de entrada (Linear Stage)
    float tau_syn[H][W] = {{ 2.0f }};  // Sinapse rápida
    float tau_mem[H][W] = {{ 10.0f }}; // Membrana lenta
    float R_mem[H][W]   = {{ 1.0f }}; 
    float v_leak[H][W]  = {{ 0.0f }};
    float v_th[H][W]    = {{ 1.5f }};  // Threshold
    float v_rst         = 0.0f;

    // Estados
    float u[H][W] = {{ 0.0f }};
    float v[H][W] = {{ 0.0f }};
    bool spikes[H][W] = {{ false }};
    
    // Entrada: Pulso unitário (será multiplicado por w_in=2.0)
    float input[H][W] = {{ 1.0f }};

    std::cout << "\n=== Teste CuBa-LIF (Current-Based LIF) ===\n";
    std::cout << "Linear -> Synapse -> Membrane with Spiking\n";
    std::cout << "Input=1.0, w_in=2.0 -> I_syn alvo=2.0. Threshold=1.5\n\n";

    for(int t=1; t<=8; t++) {
        cuba_lif_neuron<float, H, W>(input, w_in, tau_syn, tau_mem, R_mem, v_leak, v_th, v_rst, dt, u, v, spikes);
        
        std::cout << "T=" << t << "ms | u(Syn): " << u[0][0] 
                  << " | v(Mem): " << v[0][0] 
                  << " | Spike: " << spikes[0][0] << "\n";
        
        // Remove estímulo após T=1 para ver relaxamento
        if(t == 1) input[0][0] = 0.0f;
    }
}

// --------------------------------------------------------
// Testbench - CuBa-LIF Multi-Dimensional (1D/2D/3D)
// --------------------------------------------------------
void cuba_lif_dimensionality_test()
{
    std::cout << "\n========================================\n";
    std::cout << "=== Teste CuBa-LIF Multi-Dimensional ===\n";
    std::cout << "========================================\n";
    
    float dt = 1.0f;
    
    // ===== TESTE 1D: Dense Layer (5 neurônios) =====
    std::cout << "\n--- 1D: Dense Layer (5 Neurônios) ---\n";
    const int N = 5;
    
    // Parâmetros uniformes para todos os neurônios
    float in_1d[N] = {2.0f, 1.5f, 3.0f, 0.5f, 2.5f};
    float w_1d[N] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float tau_s_1d[N] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float tau_m_1d[N] = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    float R_1d[N] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float v_leak_1d[N] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float v_th_1d[N] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float v_rst_1d = 0.0f;
    
    // Estados
    float u_1d[N] = {0};
    float v_1d[N] = {0};
    bool spk_1d[N] = {false};
    
    // Simula 3 passos
    std::cout << "Input: [";
    for(int i=0; i<N; i++) std::cout << in_1d[i] << (i<N-1 ? ", " : "");
    std::cout << "]\n\n";
    
    for(int t=1; t<=3; t++) {
        cuba_lif_neuron<float, N>(in_1d, w_1d, tau_s_1d, tau_m_1d, R_1d, 
                                    v_leak_1d, v_th_1d, v_rst_1d, dt, u_1d, v_1d, spk_1d);
        
        std::cout << "Step " << t << ":\n";
        for(int i=0; i<N; i++) {
            std::cout << "  Neuron[" << i << "]: v=" << v_1d[i] 
                      << " | spike=" << spk_1d[i] << "\n";
        }
        
        // Remove entrada após primeiro passo
        if(t == 1) for(int i=0; i<N; i++) in_1d[i] = 0.0f;
    }
    
    // ===== TESTE 2D: Imagem 3x3 =====
    std::cout << "\n--- 2D: Imagem 3x3 (Conv1D ou Spatial) ---\n";
    const int H2 = 3;
    const int W2 = 3;
    
    // Entrada: Padrão de cruz
    float in_2d[H2][W2] = {
        {0.0f, 2.0f, 0.0f},
        {2.0f, 3.0f, 2.0f},
        {0.0f, 2.0f, 0.0f}
    };
    
    // Parâmetros (homogêneos)
    float w_2d[H2][W2], tau_s_2d[H2][W2], tau_m_2d[H2][W2];
    float R_2d[H2][W2], v_leak_2d[H2][W2], v_th_2d[H2][W2];
    
    for(int i=0; i<H2; i++) {
        for(int j=0; j<W2; j++) {
            w_2d[i][j] = 1.0f;
            tau_s_2d[i][j] = 2.0f;
            tau_m_2d[i][j] = 8.0f;
            R_2d[i][j] = 1.0f;
            v_leak_2d[i][j] = 0.0f;
            v_th_2d[i][j] = 2.5f;  // Threshold mais alto
        }
    }
    float v_rst_2d = 0.0f;
    
    // Estados
    float u_2d[H2][W2] = {{0}};
    float v_2d[H2][W2] = {{0}};
    bool spk_2d[H2][W2] = {{false}};
    
    std::cout << "Input Pattern (Cruz):\n";
    for(int i=0; i<H2; i++) {
        std::cout << "  [";
        for(int j=0; j<W2; j++) {
            std::cout << in_2d[i][j] << (j<W2-1 ? " " : "");
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
    
    // Simula 2 passos
    for(int t=1; t<=2; t++) {
        cuba_lif_neuron<float, H2, W2>(in_2d, w_2d, tau_s_2d, tau_m_2d, R_2d,
                                         v_leak_2d, v_th_2d, v_rst_2d, dt, u_2d, v_2d, spk_2d);
        
        std::cout << "Step " << t << " - Spikes:\n";
        for(int i=0; i<H2; i++) {
            std::cout << "  [";
            for(int j=0; j<W2; j++) {
                std::cout << spk_2d[i][j] << (j<W2-1 ? " " : "");
            }
            std::cout << "] ";
            
            // Mostra voltagens
            std::cout << " v=[";
            for(int j=0; j<W2; j++) {
                std::cout << v_2d[i][j];
                if(j<W2-1) std::cout << " ";
            }
            std::cout << "]\n";
        }
        
        // Zera entrada
        if(t == 1) {
            for(int i=0; i<H2; i++)
                for(int j=0; j<W2; j++)
                    in_2d[i][j] = 0.0f;
        }
    }
    
    // ===== TESTE 3D: Conv2D com 2 Canais 2x2 =====
    std::cout << "\n--- 3D: Conv2D (2 Canais, 2x2) ---\n";
    const int CH = 2;
    const int H3 = 2;
    const int W3 = 2;
    
    // Entrada: Canal 0 forte, Canal 1 fraco
    float in_3d[CH][H3][W3] = {
        {{3.0f, 2.5f}, {2.5f, 3.0f}},  // Canal 0: Forte
        {{0.5f, 0.5f}, {0.5f, 0.5f}}   // Canal 1: Fraco
    };
    
    // Parâmetros
    float w_3d[CH][H3][W3], tau_s_3d[CH][H3][W3], tau_m_3d[CH][H3][W3];
    float R_3d[CH][H3][W3], v_leak_3d[CH][H3][W3], v_th_3d[CH][H3][W3];
    
    for(int c=0; c<CH; c++) {
        for(int i=0; i<H3; i++) {
            for(int j=0; j<W3; j++) {
                w_3d[c][i][j] = 1.0f;
                tau_s_3d[c][i][j] = 2.0f;
                tau_m_3d[c][i][j] = 10.0f;
                R_3d[c][i][j] = 1.0f;
                v_leak_3d[c][i][j] = 0.0f;
                v_th_3d[c][i][j] = 2.0f;
            }
        }
    }
    float v_rst_3d = 0.0f;
    
    // Estados
    float u_3d[CH][H3][W3] = {{{0}}};
    float v_3d[CH][H3][W3] = {{{0}}};
    bool spk_3d[CH][H3][W3] = {{{false}}};
    
    std::cout << "Input (2 Canais, Canal 0 forte, Canal 1 fraco):\n";
    for(int c=0; c<CH; c++) {
        std::cout << "  Canal " << c << ":\n";
        for(int i=0; i<H3; i++) {
            std::cout << "    [";
            for(int j=0; j<W3; j++) {
                std::cout << in_3d[c][i][j] << (j<W3-1 ? " " : "");
            }
            std::cout << "]\n";
        }
    }
    std::cout << "\n";
    
    // Simula 3 passos
    for(int t=1; t<=3; t++) {
        cuba_lif_neuron<float, CH, H3, W3>(in_3d, w_3d, tau_s_3d, tau_m_3d, R_3d,
                                             v_leak_3d, v_th_3d, v_rst_3d, dt, 
                                             u_3d, v_3d, spk_3d);
        
        std::cout << "Step " << t << ":\n";
        for(int c=0; c<CH; c++) {
            std::cout << "  Canal " << c << " - Spikes: [";
            for(int i=0; i<H3; i++) {
                for(int j=0; j<W3; j++) {
                    std::cout << spk_3d[c][i][j] << (i<H3-1 || j<W3-1 ? " " : "");
                }
            }
            std::cout << "] v_avg=" 
                      << (v_3d[c][0][0] + v_3d[c][0][1] + v_3d[c][1][0] + v_3d[c][1][1]) / 4.0f
                      << "\n";
        }
        
        // Zera entrada
        if(t == 1) {
            for(int c=0; c<CH; c++)
                for(int i=0; i<H3; i++)
                    for(int j=0; j<W3; j++)
                        in_3d[c][i][j] = 0.0f;
        }
    }
    
    std::cout << "\n=== Teste Multi-Dimensional Completo! ===\n";
    std::cout << "As 3 versões (1D, 2D, 3D) foram executadas com sucesso.\n";
    std::cout << "O compilador selecionou automaticamente a função correta.\n";
}

int main() {
    
    sum_pooling_test();
    avg_pooling_test();
    conv2d_test();
    integrate_and_fire_test();
    affine_test();
    linear_test();
    integrator_test();
    leaky_integrator_test();
    lif_test();
    cuba_li_test();
    cuba_lif_test();
    cuba_lif_dimensionality_test();  // Novo teste multi-dimensional
    return 0;
}