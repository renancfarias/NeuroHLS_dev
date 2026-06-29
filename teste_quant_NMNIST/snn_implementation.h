#pragma once

#include "neuro_hls_functions/bit_type.h"
#include "quantization.h"

void snn_to_hls(input_t (&input)[2][34][34], bit_t (&output)[10], bool reset_potentials);