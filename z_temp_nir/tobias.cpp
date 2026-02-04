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

/**
 * Integrate-and-Fire (IF) Neuron para Vitis HLS
 * ----------------------------------------------------------------
 * Implementa a dinâmica:
 * 1. Integração: V_membrana += Input * R
 * 2. Disparo:    Se V_membrana >= Threshold -> Spike = 1, V_membrana = V_reset
 * Senão                      -> Spike = 0, V_membrana mantém valor
 * * T_DATA:  Tipo do dado de entrada/potencial (float, ap_fixed, etc.)
 * T_PARAM: Tipo dos parâmetros R e Threshold
 * H, W:    Dimensões da camada (Altura, Largura)
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
            
            // 1. Passo "Input * R" (conforme pedido)
            T_DATA injection = input[i][j] * R[i][j];
            
            // 2. Passo "Integrate" (Acumula no estado atual)
            T_DATA current_v = membrane_potential[i][j] + injection;
            
            // 3. Passo "Fire" (Comparação com T e Lógica da Imagem)
            if (current_v >= threshold[i][j]) {
                // Disparo!
                output_spikes[i][j] = true;
                membrane_potential[i][j] = v_reset; // Reset conforme imagem
            } else {
                // Sem disparo
                output_spikes[i][j] = false;
                membrane_potential[i][j] = current_v; // Mantém a carga acumulada
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
void affine_layer(
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

/**
 * Integrator Component para Vitis HLS
 * ----------------------------------------------------------------
 * Dinâmica: v_new = v_old + (Input * R)
 * * T_DATA: Tipo do dado de entrada e voltagem (float, ap_fixed)
 * T_PARAM: Tipo do parâmetro R
 * H, W: Dimensões da matriz de neurônios/entradas
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
    // Diretivas de otimização
    // #pragma HLS INLINE off
    
    // Loop sobre a altura
    loop_h: for(int i = 0; i < H; i++) {
        
        // Loop sobre a largura
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            
            // Leitura do estado anterior
            T_DATA v_old = voltage_state[i][j];
            
            // Cálculo do incremento: dV = I * R
            // Assumindo dt = 1 passo de simulação
            T_DATA dv = input[i][j] * R[i][j];
            
            // Atualização do estado (Integração)
            voltage_state[i][j] = v_old + dv;
        }
    }
}

/**
 * Leaky Integrator (LI) para Vitis HLS
 * ----------------------------------------------------------------
 * Dinâmica: tau * dv/dt = (v_leak - v) + R*I
 * Discretização (Euler): v_new = v_old + (dt/tau) * ((v_leak - v_old) + R*I)
 *
 * T_DATA: Tipo de dado (float, ap_fixed)
 * H, W: Dimensões da camada
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
            
            // 1. Leitura do estado anterior
            T_DATA v_old = v_state[i][j];
            
            // 2. Cálculo do termo de Vazamento (Decay term)
            // (Para onde a voltagem quer ir naturalmente)
            T_DATA leak_term = v_leak[i][j] - v_old;
            
            // 3. Cálculo do termo de Entrada (Input term)
            T_DATA input_term = input[i][j] * R[i][j];
            
            // 4. Cálculo da Derivada total (dv/dt)
            // dv = (1/tau) * (Leak + Input)
            T_DATA dv_dt = (leak_term + input_term) / tau[i][j];
            
            // 5. Integração de Euler
            // v_new = v_old + dv * dt
            v_state[i][j] = v_old + (dv_dt * dt);
        }
    }
}

/**
 * LIF Neuron (Leaky Integrate-and-Fire)
 * ----------------------------------------------------------------
 * Reutiliza a física do Leaky Integrator e adiciona a descontinuidade do disparo.
 * * Sequência de Operação:
 * 1. Integração (LI): Atualiza v_state baseado na corrente e vazamento.
 * 2. Comparação: Verifica se v_state >= v_threshold.
 * 3. Reset: Se disparar, v_state = v_reset. Senão, mantém o valor integrado.
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
    // 1. Passo de Integração (Reutilização do Código Anterior)
    // O compilador HLS tentará fazer "inline" disso para otimizar os loops
    #pragma HLS INLINE
    leaky_integrator<T_DATA, H, W>(input, tau, R, v_leak, dt, v_state);

    // 2. Passo de Disparo e Reset (Thresholding)
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            
            // Verifica o limiar 
            if (v_state[i][j] >= v_threshold[i][j]) {
                // DISPARO!
                spikes_out[i][j] = true;
                
                // Hard Reset: A voltagem é forçada para v_reset instantaneamente
                v_state[i][j] = v_reset; 
            } else {
                // SILÊNCIO
                spikes_out[i][j] = false;
                // v_state mantém o valor calculado pelo leaky_integrator
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

/**
 * CuBa-LI (Current-Based Leaky Integrator)
 * ----------------------------------------------------------------
 * Modelo de dois estágios:
 * 1. Synapse (u): Integra a corrente de entrada 'i' com peso 'w_in'.
 * 2. Soma (v): Integra a corrente sináptica 'u' na membrana.
 *
 * * Parâmetros:
 * tau_syn: Constante de tempo da sinapse
 * tau_mem: Constante de tempo da membrana
 * R: Resistência da membrana
 * v_leak: Tensão de vazamento
 * w_in: Peso de entrada da corrente
 */
template <
    typename T_DATA,
    int H, int W
>
void cuba_li_neuron(
    // Entradas
    const T_DATA input[H][W],     // Corrente i(t)
    
    // Parâmetros Sinápticos
    const T_DATA tau_syn[H][W],
    const T_DATA w_in[H][W],      // Atua como o "R" da sinapse
    
    // Parâmetros de Membrana
    const T_DATA tau_mem[H][W],
    const T_DATA R[H][W],
    const T_DATA v_leak[H][W],
    
    const T_DATA dt,

    // Estados (Memória)
    T_DATA u_state[H][W],         // Estado da Corrente Sináptica (u)
    T_DATA v_state[H][W]          // Estado da Tensão de Membrana (v)
) {
    #pragma HLS INLINE
    
    // Matriz de Zeros para o v_leak da sinapse (pois u tende a 0)
    // Em HLS, isso deve ser otimizado como constante se possível
    T_DATA zero_leak[H][W];
    #pragma HLS ARRAY_PARTITION variable=zero_leak complete dim=0
    init_zeros: for(int i=0; i<H; i++) 
        for(int j=0; j<W; j++) 
            zero_leak[i][j] = 0;

    // ---------------------------------------------------------
    // ESTÁGIO 1: Dinâmica da Sinapse (u)
    // Equação: tau_syn * du/dt = -u + w_in * input
    // Mapeamento para Leaky Integrator:
    // Input -> input
    // R     -> w_in (O ganho de entrada é aplicado aqui)
    // Tau   -> tau_syn
    // Leak  -> 0 (Synapse relaxa para zero)
    // ---------------------------------------------------------
    leaky_integrator<T_DATA, H, W>(input, tau_syn, w_in, zero_leak, dt, u_state);

    // ---------------------------------------------------------
    // ESTÁGIO 2: Dinâmica da Membrana (v)
    // Equação: tau_mem * dv/dt = (v_leak - v) + R * u
    // Mapeamento para Leaky Integrator:
    // Input -> u_state (A saída da sinapse é a entrada da membrana)
    // R     -> R (Resistência da membrana)
    // Tau   -> tau_mem
    // Leak  -> v_leak
    // ---------------------------------------------------------
    leaky_integrator<T_DATA, H, W>(u_state, tau_mem, R, v_leak, dt, v_state);
}

/**
 * Current-Based Leaky Integrate-and-Fire (CuBa-LIF)
 * ----------------------------------------------------------------
 * Dinâmica:
 * 1. Synapse (u): tau_syn * du/dt = -u + w_in * I
 * 2. Membrane (v): tau_mem * dv/dt = (v_leak - v) + R * u
 * 3. Fire: Se v >= Threshold -> Spike & Reset
 * * Pipeline: Input -> [Linear] -> [LI Synapse] -> [LIF Membrane] -> Spikes
 */
template <
    typename T_DATA,
    int H, int W
>
void cuba_lif_neuron(
    // Entrada
    const T_DATA input[H][W],         // Corrente externa i(t)

    // Parâmetros Sinápticos (Stage 1 & 2)
    const T_DATA w_in[H][W],          // Peso de entrada (Linear Stage)
    const T_DATA tau_syn[H][W],       // Constante de tempo da sinapse

    // Parâmetros de Membrana (Stage 3)
    const T_DATA tau_mem[H][W],       // Constante de tempo da membrana
    const T_DATA R_mem[H][W],         // Resistência da membrana
    const T_DATA v_leak[H][W],        // Potencial de vazamento
    const T_DATA v_threshold[H][W],   // Limiar de disparo
    const T_DATA v_reset,             // Tensão de reset

    const T_DATA dt,                  // Passo de tempo

    // Estados (Memória)
    T_DATA u_state[H][W],             // Estado da Sinapse (u)
    T_DATA v_state[H][W],             // Estado da Membrana (v)
    bool spikes_out[H][W]             // Saída de Spikes
) {
    #pragma HLS INLINE // Permite otimização cruzada entre as funções

    // Buffer Intermediário: Corrente ponderada (w_in * I)
    T_DATA weighted_input[H][W];
    
    // Constantes Auxiliares
    T_DATA zero_leak[H][W];     // Sinapse decai para 0
    T_DATA unit_R_syn[H][W];    // LI da sinapse tem R=1 (pois Linear já aplicou w_in)
    
    #pragma HLS ARRAY_PARTITION variable=zero_leak complete dim=0
    
    // Inicialização de constantes para reutilização das primitivas
    init_loop: for(int i=0; i<H; i++) {
        #pragma HLS UNROLL
        for(int j=0; j<W; j++) {
            zero_leak[i][j] = 0;
            unit_R_syn[i][j] = 1;
        }
    }

    // ---------------------------------------------------------
    // ESTÁGIO 1: Linear (Aplicação de w_in) 
    // ---------------------------------------------------------
    // Aplica w_in * input. Tratamos H*W como um vetor linear para simplificar.
    linear_scale<T_DATA, H*W>((T_DATA*)input, (T_DATA*)w_in, (T_DATA*)weighted_input);

    // ---------------------------------------------------------
    // ESTÁGIO 2: LI (Dinâmica da Sinapse 'u') 
    // ---------------------------------------------------------
    // Eq: tau_syn * du = -u + (1 * weighted_input)
    leaky_integrator<T_DATA, H, W>(
        weighted_input, // Entrada já ponderada
        tau_syn,        // Tau da sinapse
        unit_R_syn,     // R=1 (Scaling feito no estágio Linear)
        zero_leak,      // Leak=0 (Decai para zero)
        dt, 
        u_state         // Estado u atualizado
    );

    // ---------------------------------------------------------
    // ESTÁGIO 3: LIF (Dinâmica da Membrana 'v' + Fire) 
    // ---------------------------------------------------------
    // Eq: tau_mem * dv = (v_leak - v) + R * u
    lif_neuron<T_DATA, H, W>(
        u_state,        // A saída da sinapse 'u' é a entrada da membrana
        tau_mem,        // Tau da membrana
        R_mem,          // Resistência da membrana
        v_leak,         // Leak da membrana
        dt, 
        v_threshold,    // Parâmetros de disparo
        v_reset, 
        v_state,        // Estado v atualizado
        spikes_out      // Spikes gerados
    );
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
    affine_layer<int, N_IN, N_OUT>(I, W, b, out);

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
// Testbench - CuBa-LIF (Current-Based LIF)
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

int main() {
    
    sum_pooling_test();
    conv2d_test();
    integrate_and_fire_test();
    affine_test();
    linear_test();
    integrator_test();
    leaky_integrator_test();
    lif_test();
    cuba_li_test();
    cuba_lif_test();
    return 0;
}