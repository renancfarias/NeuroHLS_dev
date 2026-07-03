#include "lif_utils.h"

/**
 * @brief Implementação da LUT em ROM estática.
 * Armazena 2^(-x) para x entre [0, 1).
 */
decay_t get_template_val(ap_uint<LUT_BITS> index) {
#pragma HLS INLINE off
    
    // Em produção, esta tabela seria carregada de um arquivo .h externo
    // ou inicializada com valores pré-calculados.
    // O HLS detectará o array 'static const' e criará uma BRAM ou Distributed ROM.
    static const decay_t template_rom[LUT_SIZE] = {
        // Exemplo simplificado (valores reais devem ser gerados externamente):
        // 2^(-0/256), 2^(-1/256), ...
        // Como não posso listar 256 números aqui, vou usar uma aproximação 
        // linear para fins de demonstração da sintaxe, mas assuma valores REAIS de exp.
        #include "exp_lut_values.h" // Imagine que este arquivo tem: 1.0, 0.997, ...
    };
    
    // Pragma para forçar implementação como ROM
    #pragma HLS BIND_STORAGE variable=template_rom type=rom_1p impl=auto

    return template_rom[index];
}

/**
 * @brief Calcula o fator de decaimento usando lógica base-2.
 * Entrada: dt (tempo decorrido)
 * Saída: fator de decaimento (0 a 1)
 */
decay_t fast_exp_decay(time_step_t dt) {
#pragma HLS INLINE off

    // 1. Converter tempo para domínio base-2 logarítmico
    // u representa o número de "meias-vidas" que passaram
    // Formato <16, 6> para permitir até 64 meias-vidas (muito tempo) e 10 bits de fração
    ap_fixed<16, 6> u = dt * TIME_SCALING_CONST;

    // 2. Extrair parte inteira (Shift/Scaling)
    // ap_uint<6> garante range 0-63
    ap_uint<6> k = u.to_int(); 

    // 3. Extrair parte fracionária para índice da LUT (Template)
    // Pegamos os bits fracionários diretamente
    // Ex: se u = 3.5, k=3, f=0.5. Queremos o índice correspondente a 0.5.
    ap_fixed<LUT_BITS, 0> f_part = u; // O cast corta a parte inteira automaticamente
    
    // Converter a fração p/ índice inteiro (0 a 255)
    // Basicamente interpretamos os bits da fração como um inteiro
    ap_uint<LUT_BITS> index = f_part.range(LUT_BITS-1, 0);

    // 4. Lookup no Template
    decay_t base_decay = get_template_val(index);

    // 5. Aplicar Scaling (Bit Shift)
    // Em hardware, decay >> k é um shifter muito barato.
    // Se k > 16, o resultado é basicamente 0.
    if (k > 15) return 0;
    
    return base_decay >> k;
}

/**
 * @brief Kernel LIF Event-Driven
 * Este bloco mantém o estado interno (v_state) entre chamadas.
 */
void lif_core(neuron_state_t &state, spike_t &spike) {
#pragma HLS INLINE off
    time_step_t dt = spike.timestamp - state.last_spike_time;
    current_t input_current = spike.amplitude;
    voltage_t v_old = state.v_mem;
    voltage_t v_rest = state.v_leak;
    voltage_t r_mem = state.r_mem;

    // Lógica do LIF
    
    // 1. Calcular para onde a tensão quer ir (V_inf)
    //voltage_t v_inf = V_REST + (input_current * R_MEMB);
    voltage_t v_inf = v_rest + (input_current * r_mem);
    
    // 2. Calcular quanto decaiu desde o último evento
    decay_t decay = fast_exp_decay(dt);

    // 3. Atualizar Tensão
    // V_new = V_inf + (V_old - V_inf) * decay
    voltage_t v_diff = v_old - v_inf;
    voltage_t v_decayed = v_diff * decay; // Multiplicação Fixed-Point
    voltage_t v_new = v_inf + v_decayed;
    state.v_mem = v_new;
}

bool threshold (neuron_state_t &state, spike_t &spike) {
    #pragma HLS INLINE off
    if (state.v_mem >= state.v_th) {
        state.v_mem = state.v_reset;
        state.last_spike_time = spike.timestamp;
        return true;
    }
    return false;
}