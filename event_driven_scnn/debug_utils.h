#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include "types.h"
#include <hls_stream.h>

#ifndef __SYNTHESIS__
    #include <fstream>
    #include <iostream>
    #include <string>
    #include <iomanip> 
#endif

/**
 * @brief Monitor de Camada (Spy) - Full Dump
 */
// ALTERAÇÃO: Adicionado 'static inline' para evitar erro de definição múltipla no linker
static inline void monitor_layer(
    hls::stream<spike_t> &in_stream,
    hls::stream<spike_t> &out_stream,
    const char* filename = "layers_output_full.txt" 
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS INLINE off 

    // O código de arquivo só existe na simulação
    #ifndef __SYNTHESIS__
        
        // Static local funciona, mas se usar 'static inline' em múltiplos arquivos, 
        // cada arquivo terá sua cópia. Para simulação simples, isso é OK.
        // Se precisar compartilhar o 'is_first_run' globalmente, precisaria mover a impl para .cpp
        // Mas para evitar complexidade agora, mantemos assim, sabendo que cada chamada cria um stream local.
        
        static bool is_first_run = true;
        std::ofstream file; 

        // Se filename vier como NULL (segurança), usa default
        const char* final_name = (filename) ? filename : "output.txt";

        if (is_first_run) {
            file.open(final_name, std::ios::out); 
            file << "type amplitude timestamp time_step batch_idx channel_idx height_idx width_idx" << std::endl;
            is_first_run = false;
        } else {
            file.open(final_name, std::ios::app); 
        }
        file << std::fixed;
    #endif

    while (true) {
        spike_t s = in_stream.read();

        #ifndef __SYNTHESIS__
            if (file.is_open()) {
                file << s.type << " " 
                     << s.amplitude << " "     
                     << s.timestamp << " "
                     << (int)s.time_step << " "
                     << s.batch_idx << " "
                     << s.channel_idx << " "
                     << s.height_idx << " "
                     << s.width_idx << "\n";
            }
        #endif

        out_stream.write(s);

        if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            break;
        }
    }

    #ifndef __SYNTHESIS__
        if (file.is_open()) {
            file.close();
        }
    #endif
}

// ALTERAÇÃO: Adicionado 'static inline'
static inline void spike_to_txt(
    spike_t s,
    const char* filename = "layers_output_full.txt" 
) {
    #ifndef __SYNTHESIS__
        static bool is_first_run = true;
        std::ofstream file; 

        const char* final_name = (filename) ? filename : "output.txt";

        if (is_first_run) {
            file.open(final_name, std::ios::out); 
            file << "type amplitude timestamp time_step batch_idx channel_idx height_idx width_idx" << std::endl;
            is_first_run = false;
        } else {
            file.open(final_name, std::ios::app); 
        }
        file << std::fixed;

        if (file.is_open()) {
            file << s.type << " " 
                << s.amplitude << " "     
                << s.timestamp << " "
                << (int)s.time_step << " "
                << s.batch_idx << " "
                << s.channel_idx << " "
                << s.height_idx << " "
                << s.width_idx << "\n";
            
            file.close();
        }
    #endif
}

#endif