#pragma once

#include "ap_int.h"
#include "neuro_hls_functions/bit_type.h"

typedef ap_fixed<16, 8> input_t;
typedef ap_fixed<24, 8, AP_RND> weight_t;
typedef ap_fixed<24, 8> potential_t;
typedef ap_fixed<32, 8, AP_RND> temporal_t;
typedef ap_ufixed<28, 1, AP_RND, AP_SAT> alpha_syn_t;
typedef ap_ufixed<28, 1, AP_RND, AP_SAT> beta_mem_t;
typedef ap_fixed<52, 12, AP_RND, AP_SAT> dynamics_t;
