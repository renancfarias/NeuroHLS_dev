#ifndef MERGE_H
#define MERGE_H

#include <hls_stream.h>
#include "types.h"

/**
 * @brief Implementação de Merge baseada em Eventos.
 * Como no modo Spiking/Event-Driven de matrizes esparsas a informação viaja em pacotes (spike_t),
 * tudo o que a camada "Merge" precisa fazer é multiplexar o tráfego dos dois streams de entrada
 * (que estão roteando no mesmo Time Step) para um único stream de saída.
 *
 * NOTA DE HW (HLS): Lemos continuamente ambas as entradas. 
 * Apenas os marcadores TYPE_END_STEP e TYPE_END_SAMPLE precisam ser coordenados ou ignorados de uma
 * das vias para evitar duplicação no fluxo de saída.
 */

inline void Merge(hls::stream<spike_t> &in1, hls::stream<spike_t> &in2, hls::stream<spike_t> &out)
{
    bool in1_done = false;
    bool in2_done = false;
    spike_t end_token;

    // Processamos toda a in1 (camada convencional) até encontrar o fim do Time Step
    while (!in1_done) {
        spike_t s = in1.read(); // Read bloqueante: obriga que a fila tenha completado o batch atual
        if (s.type == TYPE_SPIKE) {
            out.write(s);
        } else {
            end_token = s; // Salva o token unificado (TYPE_END_STEP ou TYPE_END_SAMPLE)
            in1_done = true;
        }
    }

    // Processamos também toda a in2 (camada recorrente)
    while (!in2_done) {
        spike_t s = in2.read(); // Read bloqueante: espera a recorrência processar seu timestep respectivo
        if (s.type == TYPE_SPIKE) {
            // Alinha timestamp e time_step com os valores vindos de in1
            s.timestamp = end_token.timestamp;
            s.time_step = end_token.time_step;
            out.write(s);
        } else {
            // Em caso de fim de passo, apenas consideramos concluído, sem sobrescrever o token que irá adiante
            in2_done = true;
        }
    }

    // Emite o token apenas após ambas terem injetados seus spikes finais do passo síncrono
    out.write(end_token);
}

#endif // MERGE_H
