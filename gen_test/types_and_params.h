#ifndef _TYPES_AND_PARAMS_H_
#define _TYPES_AND_PARAMS_H_

#include "ap_fixed.h"

#include "neuro_hls_functions/bit_type.h"

typedef ap_fixed<16, 8> potential_t;
typedef ap_fixed<16, 8> input_t;

#define STEP_COUNT 10
#define DIM_1 784

#endif