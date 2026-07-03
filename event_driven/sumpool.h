#ifndef SUMPOOL_H
#define SUMPOOL_H

#include "types.h"
#include <hls_stream.h>


/**
 * @brief Sum Pooling 2D (Non-overlapping)
 * Soma a atividade dentro de uma janela KxK e emite um único valor.
 * * @tparam W_IN     Largura da Entrada
 * @tparam H_IN     Altura da Entrada
 * @tparam C        Número de Canais (Entrada == Saída no Pooling)
 * @tparam K_SIZE   Tamanho da Janela de Pooling (ex: 2 para 2x2)
 * @tparam W_OUT    Largura Saída = W_IN / K_SIZE
 * @tparam H_OUT    Altura Saída  = H_IN / K_SIZE
 */
template<
    int W_IN, int H_IN, int C,
    int K_SIZE,
    int W_OUT, int H_OUT
>
void sumpool2d_layer(
    hls::stream<spike_t> &input_stream,
    hls::stream<spike_t> &output_stream
) {
    // -------------------------------------------------------------------------
    // 1. MEMÓRIA DE ESTADO
    // -------------------------------------------------------------------------
    // Armazena a soma dos spikes na grade reduzida.
    static accum_t acc_buffer[H_OUT][W_OUT][C];

    // Particionamento cíclico nos canais para permitir processamento paralelo
    // se o pipeline desenrolar loops internos em outras camadas.
    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=cyclic factor=4 dim=3

    // -------------------------------------------------------------------------
    // 2. INICIALIZAÇÃO
    // -------------------------------------------------------------------------
    init_y: for (int y = 0; y < H_OUT; y++) {
        init_x: for (int x = 0; x < W_OUT; x++) {
            #pragma HLS PIPELINE II=1
            for (int c = 0; c < C; c++) {
                acc_buffer[y][x][c] = 0; // Pooling não tem bias, inicia com 0
            }
        }
    }

    bool is_last = false;
    time_step_t batch_time = 0;

    // -------------------------------------------------------------------------
    // 3. PROCESSAMENTO (Redução de Coordenadas)
    // -------------------------------------------------------------------------
    process_loop: while (!is_last) {
        #pragma HLS PIPELINE II=1

        spike_t s = input_stream.read();

        if (s.last_feature) {
            is_last = true;
            batch_time = s.timestamp;
        } 
        else {
            // Decodificação do Índice Flat
            int flat_idx = s.index;
            int in_c = flat_idx % C;
            int temp = flat_idx / C;
            int in_x = temp % W_IN;
            int in_y = temp / W_IN;
            
            // --- CÁLCULO DA COORDENADA DE SAÍDA ---
            // Como é Input Stationary e Stride=Kernel (não sobreposto),
            // a coordenada de saída é apenas uma divisão inteira.
            // Ex: Em 2x2, pixel 0 e 1 mapeiam para 0. Pixel 2 e 3 mapeiam para 1.
            
            // O Vitis HLS otimiza divisões por constante (potência de 2 vira shift)
            int out_y = in_y / K_SIZE;
            int out_x = in_x / K_SIZE;

            // Segurança: Verifica se está dentro dos limites (útil se W_IN não for múltiplo de K)
            if (out_y < H_OUT && out_x < W_OUT) {
                // Acumula a amplitude no "super-pixel" correspondente
                acc_buffer[out_y][out_x][in_c] += s.amplitude;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. WRITE BACK (Disparo da Saída)
    // -------------------------------------------------------------------------
    write_y: for (int y = 0; y < H_OUT; y++) {
        write_x: for (int x = 0; x < W_OUT; x++) {
            write_c: for (int c = 0; c < C; c++) {
                #pragma HLS PIPELINE II=1
                
                accum_t potential = acc_buffer[y][x][c];
                
                if (potential != 0) {
                    spike_t out_s;
                    // Recalcula Flat Index para a dimensão de saída
                    out_s.index = ((y * W_OUT) + x) * C + c;
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