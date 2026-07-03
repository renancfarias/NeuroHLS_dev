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
const time_step_t LIF_DT = 1e-4;
const time_step_t LIF_BETA = 0.001;
const time_step_t LIF_TAU = (double)LIF_DT / (1.0 - (double)LIF_BETA);
const voltage_t LIF_V_REST = 0.0;
const voltage_t LIF_V_THRESH = 1.0;
const voltage_t LIF_V_RESET = 0.0;
const voltage_t LIF_R_MEMB = LIF_TAU / LIF_DT;
const accum_t LIF_LN2 = 0.69314718056;

// Constante de Escala de Tempo (Pré-calculada)
// static const garante visibilidade interna sem conflito de linker
static const ap_fixed<32, 16> LIF_TIME_SCALING_CONST = (accum_t) 1.0 / (LIF_TAU * LIF_LN2);

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

typedef decay_t (*lif_decay_func_t)(time_step_t dt);

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

/**
 * @brief Kernel Integrate-and-Fire (Sem Leak)
 * O timestamp serve apenas para registrar quando o evento ocorreu.
 */
void if_core(neuron_state_t &state, spike_t &spike);

/**
 * @brief Verifica o limiar com soft reset.
 */
bool threshold_check(neuron_state_t &state, spike_t &spike);

#endif // LIF_UTILS_H
