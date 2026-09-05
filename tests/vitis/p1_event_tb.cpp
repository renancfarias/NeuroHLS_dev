#include <cassert>

#include "p1_event_top.h"

int main()
{
    hls::stream<ed_spike_t> input;
    hls::stream<ed_spike_t> output;

    ed_spike_t spike = {};
    spike.type = ED_TYPE_SPIKE;
    spike.amplitude = 3;
    spike.channel_idx = 0;
    spike.height_idx = 1;
    spike.width_idx = 1;
    input.write(spike);

    ed_spike_t marker = {};
    marker.type = ED_TYPE_END_STEP;
    marker.time_step = 4;
    input.write(marker);

    p1_event_top(input, output);

    ed_spike_t result = output.read();
    assert(result.type == ED_TYPE_SPIKE);
    assert(result.amplitude == ed_current_t(6));
    assert(result.height_idx == 1);
    assert(result.width_idx == 1);

    ed_spike_t end = output.read();
    assert(end.type == ED_TYPE_END_STEP);
    assert(end.time_step == 4);
    assert(output.empty());
    return 0;
}
