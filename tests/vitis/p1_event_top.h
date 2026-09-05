#pragma once

#include "../../neuro_hls/backend/neuro_hls_functions/event_driven.h"

void p1_event_top(
    hls::stream<ed_spike_t>& input,
    hls::stream<ed_spike_t>& output);
