#ifndef FLATTEN_H
#define FLATTEN_H

#include <hls_stream.h>
#include "types.h"

// Flatten Layer: Transforma (C, H, W) -> (C*H*W)
template<int C_IN, int H_IN, int W_IN>
void flatten_layer(
    hls::stream<spike_t> &input_stream,
    hls::stream<spike_t> &output_stream
) {
    
    //#pragma HLS INLINE off
    bool end_of_processing = false;

    main_loop: while (!end_of_processing) {
        #pragma HLS PIPELINE
        spike_t s = input_stream.read();

        
            // Cálculo do índice linear (Channel First / NCHW)
            // idx = (c * H * W) + (y * W) + x
            int linear_idx = (s.channel_idx * (H_IN * W_IN)) + 
                                    (s.height_idx * W_IN) + 
                                    s.width_idx;

            spike_t out_s = s;
            // O novo índice linear é colocado no comprimento
            out_s.channel_idx = 0;
            out_s.height_idx = 0;
            out_s.width_idx = linear_idx;
            
            output_stream.write(out_s);
        
        if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE)
            end_of_processing = true;

    }
}

#endif