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

int main() {
    
    sum_pooling_test();
    conv2d_test();
    integrate_and_fire_test();
    affine_test();
    return 0;
}