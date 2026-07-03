#ifndef TYPES_H
#define TYPES_H

#include <ap_fixed.h>
#include <hls_stream.h>

// Definição de Tipos de Ponto Fixo
// <W, I>: W = largura total, I = bits inteiros
// Range: [-8388608, 8388607.99998], precisao: ~1.5e-5
typedef ap_fixed<40, 24> voltage_t;
typedef ap_fixed<16, 8> current_t;
typedef ap_fixed<24, 8> time_step_t;
typedef ap_fixed<16, 8> threshold_t;

// Tipo interno para o fator de decaimento (sempre entre 0 e 1)
// Usamos unsigned para maior precisão na parte fracionária
typedef ap_fixed<24, 8> tau_t;
typedef ap_fixed<24, 8> decay_t;
typedef ap_fixed<16, 8> weight_t; // Novo tipo para pesos

// --- NOVO TIPO: ACUMULADOR ---
// Precisamos de mais bits para somar várias multiplicações sem estourar.
// <40, 24> dá 24 bits inteiros e 16 fracionários.
typedef ap_fixed<40, 24> accum_t;

typedef struct {
    ap_uint<4> type;
    current_t amplitude;
    time_step_t timestamp;
    ap_uint<32> time_step;
    ap_uint<32> batch_idx;
    ap_uint<32> channel_idx;
    ap_uint<32> height_idx;
    ap_uint<32> width_idx;
    
} spike_t;


// const int NUM_INPUTS = 784;
const int NUM_OUTPUTS = 10;

// Definição de constantes para o campo 'type'
// Usamos char (8 bits) para controle
const int TYPE_SPIKE = 0;     // Dado válido
const int TYPE_END_STEP = 1;  // Fim do Time Step
const int TYPE_END_SAMPLE = 2; // Fim da amostra

const int FIFO_DEPTH = 16; // Profundidade dos buffers de stream (ajustável)

#endif
