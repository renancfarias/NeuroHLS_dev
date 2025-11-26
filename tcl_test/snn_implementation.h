#ifndef SNN_IMPLEMENTATION_H_
#define SNN_IMPLEMENTATION_H_

#include "types_and_params.h"
#include "neuro_hls_functions/bit_type.h"
#include "neuro_hls_functions/dense.h"

void snn_to_hls(input_t input[784], bit_t output[10]);

#endif