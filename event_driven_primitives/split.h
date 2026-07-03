#ifndef SPLIT_H
#define SPLIT_H

#include <hls_stream.h>
#include "types.h"

/**
 * @brief Implementação de Split (Demux / Broadcast) baseada em Eventos.
 * 
 * Esta função consome pacotes (spikes e tokens de sincronização) 
 * de uma única stream de entrada (FIFO) e faz o broadcast (cópia) 
 * exato desse pacote para duas streams de saída em paralelo.
 * 
 * Muito útil para arquiteturas de rede recorrentes (onde a saída de 
 * uma camada vai para a próxima MAS TAMBÉM retorna para o início) 
 * ou conexões residuais (Skip Connections).
 *
 * @param in  Stream de entrada (Eventos/Spikes a serem consumidos)
 * @param out1 Primeira stream de saída (ex: caminho contínuo da rede)
 * @param out2 Segunda stream de saída (ex: caminho recorrente ou residual)
 */
inline void Split(hls::stream<spike_t> &in, hls::stream<spike_t> &out1, hls::stream<spike_t> &out2)
{
    bool step_done = false;

    // Lemos a stream de entrada ininterruptamente até recebermos
    // o token de fim daquela amostra (TYPE_END_SAMPLE).
    while (!step_done)
    {
        if (!in.empty())
        {
            // Lê o pacote apenas uma vez
            spike_t s = in.read();
            
            // Duplica (escreve) o pacote integralmente em ambas as saídas
            out1.write(s);
            out2.write(s);
            
            // Se foi o último pacote, encerra o laço
            if (s.type == TYPE_END_STEP ||s.type == TYPE_END_SAMPLE)
            {
                step_done = true;
            }
        }
    }
}

#endif // SPLIT_H
