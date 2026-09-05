#include "p1_event_top.h"

void p1_event_top(
    hls::stream<ed_spike_t>& input,
    hls::stream<ed_spike_t>& output)
{
    static const ed_weight_t weights[1][1][1][1] = {{{{2}}}};
    Conv2d<1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1>(
        input, output, weights);
}
