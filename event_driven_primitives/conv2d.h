#ifndef CONV2D_H
#define CONV2D_H

#include <hls_stream.h>
#include "types.h"

// Estrutura para fila de coordenadas ativas (apenas Y e X, pois C é totalmente paralelo)
struct CoordYX {
    ap_uint<16> y;
    ap_uint<16> x;
};

template<
    int W_IN, int H_IN, int C_IN, int C_OUT,
    int K_H,  int K_W, int S_H,  int S_W, int P_H,  int P_W
>
void conv2d_layer_no_bias(
    hls::stream<spike_t> &input_stream,
    hls::stream<spike_t> &output_stream,
    const weight_t weights[C_OUT][C_IN][K_H][K_W]
) {
    constexpr int W_OUT = (W_IN - K_W + 2 * P_W) / S_W + 1;
    constexpr int H_OUT = (H_IN - K_H + 2 * P_H) / S_H + 1;
    
    // Tamanho máximo da fila = número total de pixels espaciais
    constexpr int NUM_SPATIAL = H_OUT * W_OUT;
    
    // 1. Buffer de Acumulação
    static accum_t acc_buffer[H_OUT][W_OUT][C_OUT];
    
    // 2. Bitmap de Dirty Flags (bit por coordenada espacial)
    static ap_uint<NUM_SPATIAL> dirty_map;
    
    // 3. FIFO de pixels ativos (Active List)
    static CoordYX active_fifo[NUM_SPATIAL];
    static int fifo_count = 0;

    bool end_of_processing = false;

    // Particionamento e Dependências
    #pragma HLS DEPENDENCE variable=acc_buffer inter false
    #pragma HLS ARRAY_RESHAPE variable=active_fifo type=complete
    #pragma HLS BIND_STORAGE variable=acc_buffer type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=complete dim=3
    // Reset Inicial (apenas uma vez ao ligar)
    static bool first_run = true;
    if (first_run) {
        init_global: for (int y = 0; y < H_OUT; y++) {
            for (int x = 0; x < W_OUT; x++) {
                #pragma HLS PIPELINE ii=1
                for (int c = 0; c < C_OUT; c++) {
                    acc_buffer[y][x][c] = 0;
                }
            }
        }
        dirty_map = 0;
        fifo_count = 0;
        first_run = false;
    }

    input_loop: while (!end_of_processing) {
        
        spike_t s = input_stream.read();

        // MODO 1: ACUMULAÇÃO (Processamento de Eventos)
        if (s.type == TYPE_SPIKE) {
            int in_c = s.channel_idx;
            int in_x = s.width_idx;
            int in_y = s.height_idx;
            current_t val = s.amplitude;
            
            conv_y: 
            for (int ky = 0; ky < K_H; ky++) {
                conv_x: 
                for (int kx = 0; kx < K_W; kx++) {
                    #pragma HLS PIPELINE ii=1
                    int num_y = in_y + P_H - ky;
                    int num_x = in_x + P_W - kx;
                    
                    if (num_y >= 0 && num_x >= 0 && (num_y % S_H == 0) && (num_x % S_W == 0)) {
                        int out_y = num_y / S_H;
                        int out_x = num_x / S_W;

                        if (out_y < H_OUT && out_x < W_OUT) {
                            int dirty_idx = out_y * W_OUT + out_x;
                            
                            // GESTÃO DA FILA DE ATIVOS
                            // Se este pixel espacial (y,x) ainda não foi tocado neste passo, adiciona à fila
                            if (!dirty_map[dirty_idx]) {
                                if (fifo_count < NUM_SPATIAL) {
                                    CoordYX coord;
                                    coord.y = out_y;
                                    coord.x = out_x;
                                    active_fifo[fifo_count++] = coord;
                                    dirty_map[dirty_idx] = 1;
                                }
                            }

                            // ATUALIZAÇÃO DOS CANAIS (Paralela)
                            filter_loop: 
                            for (int cout = 0; cout < C_OUT; cout++) {
                                weight_t w = weights[cout][in_c][ky][kx];
                                acc_buffer[out_y][out_x][cout] += val * w;
                     
                            }
                        }
                    }
                }
            }
        }
        // MODO 2: FLUSH OTIMIZADO (Apenas Ativos)
        else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            // Itera apenas sobre a lista de coordenadas ativas
            flush_loop: 
            for (int i = 0; i < fifo_count; i++) {
                CoordYX coord = active_fifo[i];
                int fy = coord.y;
                int fx = coord.x;
                // Processa todos os canais desta coordenada espacial
                flush_loop_channel:
                for (int c = 0; c < C_OUT; c++) {
                    #pragma HLS PIPELINE ii=1
                    
                    accum_t potential = acc_buffer[fy][fx][c];
  
                    spike_t out_s;
                    out_s.type = TYPE_SPIKE;
                    out_s.amplitude = potential; 
                    out_s.height_idx = fy;
                    out_s.width_idx = fx;
                    out_s.channel_idx = c;
                    out_s.timestamp = s.timestamp;
                    out_s.time_step = s.time_step;
                    out_s.batch_idx = s.batch_idx;
                    output_stream.write(out_s);
                    
                    // Reset do acumulador
                    acc_buffer[fy][fx][c] = 0;
                }

                // Limpa a flag de dirty para o próximo passo
                int dirty_idx = fy * W_OUT + fx;
                dirty_map[dirty_idx] = 0;
            }
            
            // Reseta contador da fila
            fifo_count = 0;
        
            output_stream.write(s); // Propaga token de fim
            end_of_processing = true;

        }
    }
}

template<
    int W_IN, int H_IN, int C_IN, int C_OUT,
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W
>
void conv2d_layer(
    hls::stream<spike_t> &input_stream,
    hls::stream<spike_t> &output_stream,
    const weight_t weights[C_OUT][C_IN][K_H][K_W],
    const weight_t biases[C_OUT]
) {
    constexpr int W_OUT = (W_IN - K_W + 2 * P_W) / S_W + 1;
    constexpr int H_OUT = (H_IN - K_H + 2 * P_H) / S_H + 1;

    static_assert(W_OUT > 0 && H_OUT > 0, "Erro: dimensoes de saida invalidas");

    static accum_t acc_buffer[H_OUT][W_OUT][C_OUT];

    #pragma HLS ARRAY_PARTITION variable=acc_buffer type=cyclic factor=C_OUT dim=3

    init_loop: for (int y = 0; y < H_OUT; y++) {
        for (int x = 0; x < W_OUT; x++) {
            #pragma HLS PIPELINE II=1
            for (int c = 0; c < C_OUT; c++) {
                acc_buffer[y][x][c] = biases[c];
            }
        }
    }

    bool end_of_processing = false;
    spike_t end_token;

    process_loop: while (!end_of_processing) {
        #pragma HLS PIPELINE II=1

        spike_t s = input_stream.read();

        if (s.type == TYPE_SPIKE) {
            int in_c = s.channel_idx;
            int in_y = s.height_idx;
            int in_x = s.width_idx;
            current_t val = s.amplitude;

            if (in_c < C_IN && in_y < H_IN && in_x < W_IN) {
                kernel_y: for (int ky = 0; ky < K_H; ky++) {
                    kernel_x: for (int kx = 0; kx < K_W; kx++) {
                        int num_y = in_y + P_H - ky;
                        int num_x = in_x + P_W - kx;

                        bool align_y = (num_y % S_H == 0);
                        bool align_x = (num_x % S_W == 0);

                        if (align_y && align_x) {
                            int out_y = num_y / S_H;
                            int out_x = num_x / S_W;

                            if (out_y >= 0 && out_y < H_OUT && out_x >= 0 && out_x < W_OUT) {
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
        } else if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            end_token = s;
            end_of_processing = true;
        }
    }

    write_y: for (int y = 0; y < H_OUT; y++) {
        write_x: for (int x = 0; x < W_OUT; x++) {
            write_c: for (int c = 0; c < C_OUT; c++) {
                #pragma HLS PIPELINE II=1

                accum_t potential = acc_buffer[y][x][c];

                if (potential != 0) {
                    spike_t out_s;
                    out_s.type = TYPE_SPIKE;
                    out_s.amplitude = (current_t)potential;
                    out_s.timestamp = end_token.timestamp;
                    out_s.time_step = end_token.time_step;
                    out_s.batch_idx = end_token.batch_idx;
                    out_s.channel_idx = c;
                    out_s.height_idx = y;
                    out_s.width_idx = x;
                    output_stream.write(out_s);
                }
            }
        }
    }

    end_token.amplitude = 0;
    end_token.channel_idx = 0;
    end_token.height_idx = 0;
    end_token.width_idx = 0;
    output_stream.write(end_token);
}
#endif
