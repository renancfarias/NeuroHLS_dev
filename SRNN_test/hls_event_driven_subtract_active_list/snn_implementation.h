#pragma once

#include "neuro_hls_functions/bit_type.h"
#include "quantization.h"
#include "neuro_hls_functions/event_driven.h"

#define NEURO_HLS_EVENT_DT 0.0001
#define NEURO_HLS_EVENT_CUBA_LIF_ACTIVE_LIST 1
#define NEURO_HLS_ACTIVE_NOISE_THRESHOLD 9.9999999999999995e-07
void snn_to_hls(hls::stream<ed_spike_t>& input_stream, hls::stream<ed_spike_t>& output_stream, bool reset_potentials);
