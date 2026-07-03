#ifndef TYPES_H
#define TYPES_H

#include <ap_fixed.h>
#include <hls_stream.h>

// Definição de Tipos de Ponto Fixo
// <W, I>: W = largura total, I = bits inteiros
// Range: [-128, 127.996], Precisão: ~0.0039
typedef ap_fixed<32, 8, AP_RND> voltage_t; 
typedef ap_fixed<32, 8, AP_RND> current_t;
typedef ap_fixed<32, 8, AP_RND> time_step_t;
typedef ap_fixed<32, 8, AP_RND> threshold_t;

// Tipo interno para o fator de decaimento (sempre entre 0 e 1)
// Usamos unsigned para maior precisão na parte fracionária
typedef ap_ufixed<32, 1, AP_RND> decay_t;
typedef ap_fixed<32, 8, AP_RND> weight_t; // Novo tipo para pesos

// --- NOVO TIPO: ACUMULADOR ---
// Precisamos de mais bits para somar várias multiplicações sem estourar.
// <32, 16> dá 16 bits inteiros e 16 fracionários.
typedef ap_fixed<64, 16, AP_RND> accum_t;

typedef struct {
    int index;
    current_t amplitude;
    time_step_t timestamp;
    bool last_feature;
    bool reset_vmem;
} spike_t;


// const int NUM_INPUTS = 784;
const int NUM_OUTPUTS = 10;

#endif