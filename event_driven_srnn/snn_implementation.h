#pragma once

#include "types.h"

void snn_to_hls(hls::stream<spike_t> &input, hls::stream<spike_t> &output);