#pragma once

#include "ap_int.h"
#include "neuro_hls_functions/bit_type.h"

typedef ap_fixed<16, 8> input_t;
typedef ap_fixed<24, 8, AP_RND> weight_t;
typedef ap_fixed<24, 8> potential_t;
typedef ap_fixed<32, 8, AP_RND> temporal_t;
typedef ap_fixed<32, 8, AP_RND> event_tau_t;
typedef ap_uint<6> event_shift_t;
typedef ap_uint<3> event_shift_count_t;
typedef ap_uint<32> event_index_t;
