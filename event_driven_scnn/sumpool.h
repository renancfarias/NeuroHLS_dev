#ifndef SUMPOOL_H
#define SUMPOOL_H

#include "types.h"
#include <hls_stream.h>

template<int C, int W_IN, int H_IN, int K_SIZE>
void sumpool2d_layer(
    hls::stream<spike_t> &input_stream,
    hls::stream<spike_t> &output_stream
) {
    constexpr int W_OUT = W_IN / K_SIZE;
    constexpr int H_OUT = H_IN / K_SIZE;
    constexpr int NUM_OUTPUTS = W_OUT * H_OUT * C; // Total de pixels de saída

    // 1. Buffer de Acumulação e Flags
    static accum_t buffer[H_OUT][W_OUT][C];
    static bool is_dirty[H_OUT][W_OUT][C];
    
    // Lista de Ativos (FIFO) para reset rápido
    // Armazena coordenadas lineares ou struct
    static int active_fifo[NUM_OUTPUTS]; 
    static int fifo_count = 0;

    // Particionamento: Pooling é "channel-independent", podemos paralelizar em C
    #pragma HLS ARRAY_PARTITION variable=buffer type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=is_dirty type=complete dim=3

    // Inicialização
    static bool first_run = true;
    if (first_run) {
        init_loop:
        for(int y=0; y<H_OUT; y++)
            for(int x=0; x<W_OUT; x++)
                for(int k=0; k<C; k++) {
                    #pragma HLS PIPELINE
                    buffer[y][x][k] = 0;
                    is_dirty[y][x][k] = false;
                }
        fifo_count = 0;
        first_run = false;
    }

    bool end_of_processing = false;

    // Loop Principal
    process_loop: while (!end_of_processing) {
        #pragma HLS PIPELINE II=1
        
        spike_t s = input_stream.read();

        // MODO 1: ACUMULAÇÃO
        if (s.type == TYPE_SPIKE) {
            int in_c = s.channel_idx;
            
            // Mapeamento de coordenadas (Pooling)
            int out_y = s.height_idx / K_SIZE;
            int out_x = s.width_idx / K_SIZE;
            
            if (out_y < H_OUT && out_x < W_OUT && in_c < C) {
                // Soma a amplitude
                buffer[out_y][out_x][in_c] += s.amplitude;

                // Adiciona à lista de ativos se for a primeira vez
                if (!is_dirty[out_y][out_x][in_c]) {
                    is_dirty[out_y][out_x][in_c] = true;
                    
                    // codifica coordenada linear para economizar muxing na fifo
                    // idx = (y * W * C) + (x * C) + c
                    int idx = (out_y * W_OUT * C) + (out_x * C) + in_c;
                    if (fifo_count < NUM_OUTPUTS) {
                        active_fifo[fifo_count++] = idx;
                    }
                }
            }
        }
        // MODO 2: EMISSÃO E RESET
        else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            
            flush_loop: for (int i = 0; i < fifo_count; i++) {
                #pragma HLS PIPELINE II=1
                
                int idx = active_fifo[i];
                
                // Decodifica coordenadas
                int c = idx % C;
                int temp = idx / C;
                int x = temp % W_OUT;
                int y = temp / W_OUT;

                // Emite spike somado
                accum_t val = buffer[y][x][c];

                spike_t out_s;
                out_s.type = TYPE_SPIKE;
                out_s.amplitude = (current_t)val;
                out_s.height_idx = y;
                out_s.width_idx = x;
                out_s.channel_idx = c;
                out_s.timestamp = s.timestamp;
                out_s.time_step = s.time_step;
                out_s.batch_idx = s.batch_idx;
                output_stream.write(out_s);
                
                // Reset
                buffer[y][x][c] = 0;
                is_dirty[y][x][c] = false;
            }
            
            fifo_count = 0;
            output_stream.write(s); // Repassa token
            end_of_processing = true;
        }
    }
}
#endif