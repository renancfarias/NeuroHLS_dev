#ifndef LIF_UTILS_H
#define LIF_UTILS_H

#include <ap_int.h>
#include <ap_fixed.h>
#include "types.h" // Garante que time_step_t, voltage_t, spike_t, etc. existam

// =========================================================
// 1. CONSTANTES E MACROS
// =========================================================

// Tamanho da LUT do Template
#define LUT_BITS 8
#define LUT_SIZE 256

// Constantes Físicas
// Usamos 'constexpr' para garantir que o compilador HLS calcule isso
// antes da síntese, evitando lógica de divisão/multiplicação no hardware.
constexpr float DT = 1e-4;
constexpr float BETA = 0.001;

// Nota: Se time_step_t for um tipo fixed-point, certifique-se que
// a precisão é suficiente para estes cálculos, ou use float/double aqui e faça cast depois.
constexpr float TAU = DT / (1.0 - BETA);

constexpr voltage_t V_REST = 0.0;
constexpr voltage_t V_THRESH = 1.0;
constexpr voltage_t V_RESET = 0.0;
constexpr voltage_t R_MEMB = TAU / DT;
constexpr accum_t LN2 = 0.69314718056;

// Constante de Escala de Tempo (Pré-calculada)
// static const garante visibilidade interna sem conflito de linker
static const double TIME_SCALING_CONST = (accum_t) 1.0 / (TAU * LN2);

// =========================================================
// 2. ESTRUTURAS DE DADOS
// =========================================================

typedef struct {
    int index;
    voltage_t v_mem;
    voltage_t v_leak;
    voltage_t v_reset;
    voltage_t r_mem;
    time_step_t tau;
    time_step_t last_spike_time;
    current_t input_current;
    threshold_t v_th;
} neuron_state_t;

// =========================================================
// 3. PROTÓTIPOS DE FUNÇÕES
// =========================================================

/**
 * @brief Implementação da LUT em ROM estática.
 * Armazena 2^(-x) para x entre [0, 1).
 */
decay_t get_template_val(ap_uint<LUT_BITS> index);

/**
 * @brief Calcula o fator de decaimento usando lógica base-2.
 * Entrada: dt (tempo decorrido)
 * Saída: fator de decaimento (0 a 1)
 */
decay_t fast_exp_decay(time_step_t dt);

/**
 * @brief Kernel LIF Event-Driven
 * Atualiza a tensão do neurônio baseado no tempo decorrido e input.
 */
void lif_core(neuron_state_t &state, spike_t &spike);

/**
 * @brief Verifica disparo
 * Se disparar, reseta a tensão e retorna true.
 */
bool threshold(neuron_state_t &state, spike_t &spike);

#endif // LIF_UTILS_H