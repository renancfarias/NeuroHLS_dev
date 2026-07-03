#ifndef CONV2D_FLEXIBLE_H
#define CONV2D_FLEXIBLE_H

#include "types.h"
#include <hls_stream.h>

/**
 * @brief Convolução 2D Flexível (Stride + Kernel Retangular + Padding Assimétrico)
 * * @tparam W_IN   Largura da Entrada
 * @tparam H_IN   Altura da Entrada
 * @tparam C_IN   Canais de Entrada
 * @tparam C_OUT  Canais de Saída
 * * @tparam K_H    Altura do Kernel (ex: 3)
 * @tparam K_W    Largura do Kernel (ex: 5)
 * * @tparam S_H    Stride Vertical (ex: 2)
 * @tparam S_W    Stride Horizontal (ex: 2)
 * * @tparam P_H    Padding Vertical (Top/Bottom)
 * @tparam P_W    Padding Horizontal (Left/Right)
 */
template<
    int W_IN, int H_IN, int C_IN, int C_OUT,
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W
>
void conv2d_layer(
    hls::stream<spike_t> &input_stream,
    hls::stream<spike_t> &output_stream,
    // Matriz de pesos ajustada para kernel retangular [C_OUT][C_IN][H][W]
    const weight_t weights[C_OUT][C_IN][K_H][K_W],
    const weight_t biases[C_OUT]
) {
    // -------------------------------------------------------------------------
    // CÁLCULO AUTOMÁTICO DAS DIMENSÕES (CONSTEXPR)
    // -------------------------------------------------------------------------
    // O compilador resolve isso antes da síntese.
    // Fórmula: floor((Input - Kernel + 2*Padding) / Stride) + 1
    constexpr int W_OUT = (W_IN - K_W + 2 * P_W) / S_W + 1;
    constexpr int H_OUT = (H_IN - K_H + 2 * P_H) / S_H + 1;

    // Verificação de segurança em tempo de compilação (Opcional, mas útil)
    static_assert(W_OUT > 0 && H_OUT > 0, "Erro: Dimensoes de saida invalidas/negativas!");

    // 1. MEMÓRIA DE ESTADO (Feature Maps de Saída)
    static accum_t acc_buffer[H_OUT][W_OUT][C_OUT];
    
    // Particionamento para acesso paralelo aos filtros de saída
    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=cyclic factor=C_OUT dim=3

    // 2. INICIALIZAÇÃO
    init_loop: for (int y = 0; y < H_OUT; y++) {
        for (int x = 0; x < W_OUT; x++) {
            #pragma HLS PIPELINE II=1
            for (int c = 0; c < C_OUT; c++) {
                acc_buffer[y][x][c] = biases[c];
            }
        }
    }

    bool is_last = false;
    time_step_t batch_time = 0;

    // 3. PROCESSAMENTO (Input Stationary com Stride)
    process_loop: while (!is_last) {
        #pragma HLS PIPELINE II=1

        spike_t s = input_stream.read();

        if (s.last_feature) {
            is_last = true;
            batch_time = s.timestamp;
        } 
        else {
            // A. Decodificação de Coordenadas (Input Flat Index -> 3D)
            int flat_idx = s.index;
            int in_c = flat_idx % C_IN;
            int temp = flat_idx / C_IN;
            int in_x = temp % W_IN;
            int in_y = temp / W_IN;
            
            current_t val = s.amplitude;

            // B. Projeção com Stride e Kernel Retangular
            // Loop sobre as dimensões do kernel (K_H x K_W)
            kernel_y: for (int ky = 0; ky < K_H; ky++) {
                kernel_x: for (int kx = 0; kx < K_W; kx++) {
                    
                    // --- A LÓGICA DO STRIDE ---
                    // A relação normal de conv é: y_in = y_out * S + ky - P
                    // Queremos achar y_out dado y_in.
                    // y_out * S = y_in + P - ky
                    
                    int num_y = in_y + P_H - ky;
                    int num_x = in_x + P_W - kx;

                    // Condição 1: Divisibilidade
                    // O pixel de entrada só contribui para a saída se cair na grade do stride
                    bool align_y = (num_y % S_H == 0);
                    bool align_x = (num_x % S_W == 0);

                    if (align_y && align_x) {
                        int out_y = num_y / S_H;
                        int out_x = num_x / S_W;

                        // Condição 2: Boundary Check
                        // Verifica se a coordenada calculada está dentro do mapa de saída
                        if (out_y >= 0 && out_y < H_OUT && out_x >= 0 && out_x < W_OUT) {
                            
                            // Atualiza todos os filtros
                            filter_loop: for (int cout = 0; cout < C_OUT; cout++) {
                                #pragma HLS UNROLL 
                                weight_t w = weights[cout][in_c][ky][kx];
                                acc_buffer[out_y][out_x][cout] += val * w;
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. WRITE BACK
    write_y: for (int y = 0; y < H_OUT; y++) {
        write_x: for (int x = 0; x < W_OUT; x++) {
            write_c: for (int c = 0; c < C_OUT; c++) {
                #pragma HLS PIPELINE II=1
                
                accum_t potential = acc_buffer[y][x][c];
                
                if (potential != 0) {
                    spike_t out_s;
                    // Recalcula Flat Index de Saída
                    out_s.index = ((y * W_OUT) + x) * C_OUT + c;
                    out_s.amplitude = (current_t)potential;
                    out_s.timestamp = batch_time;
                    out_s.last_feature = false;
                    output_stream.write(out_s);
                }
            }
        }
    }

    // Token Final
    spike_t end_token;
    end_token.index = 0;
    end_token.amplitude = 0;
    end_token.timestamp = batch_time;
    end_token.last_feature = true;
    output_stream.write(end_token);
}

#endif