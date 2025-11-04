#ifndef _TYPES_AND_PARAMS_HPP_
#define _TYPES_AND_PARAMS_HPP_

#include <ap_fixed.h>
#include <ap_int.h>

#define NUM_INPUTS 784
#define NUM_OUTPUTS 10

#define TOTAL_SAMPLES 9984

#define NUM_STEPS 100
#define NUM_SAMPLES 1//2910

#define THRESHOLD 1.0
#define DECAY 0.7

#define input_t ap_fixed<16, 8>
#define weight_t ap_fixed<16, 8>
#define potential_t ap_fixed<16, 8>
#define bit_t ap_uint<1>

namespace layer {
    const ap_fixed<16, 8> decay = DECAY;
    const ap_fixed<16, 8> threshold = THRESHOLD;
}

#endif // _TYPES_AND_PARAMS_HPP_