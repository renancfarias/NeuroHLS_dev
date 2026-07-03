#ifndef NETWORK_H
#define NETWORK_H

#include <hls_stream.h>
#include "types.h"

void snn(hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes);

void scnn (hls::stream<spike_t> &input_spikes, hls::stream<spike_t> &output_spikes);

#endif