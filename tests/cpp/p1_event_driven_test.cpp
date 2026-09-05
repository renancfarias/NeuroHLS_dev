#include <cassert>
#include <cmath>

#include "neuro_hls_functions/time_driven.h"
#include "neuro_hls_functions/event_driven.h"

static bool close(double left, double right) {
    return std::fabs(left - right) < 1e-5;
}

static ed_spike_t event(int c, int h, int w, double amplitude) {
    ed_spike_t spike = {};
    spike.type = ED_TYPE_SPIKE;
    spike.amplitude = amplitude;
    spike.channel_idx = c;
    spike.height_idx = h;
    spike.width_idx = w;
    return spike;
}

static ed_spike_t marker(int type, int step) {
    ed_spike_t spike = {};
    spike.type = type;
    spike.timestamp = step;
    spike.time_step = step;
    spike.batch_idx = 7;
    return spike;
}

static int drain_window(
    hls::stream<ed_spike_t>& output,
    int expected_marker_type,
    ed_spike_t* first_spike = 0,
    ed_spike_t* end_marker = 0)
{
    int spike_count = 0;
    while (true) {
        assert(!output.empty());
        ed_spike_t result = output.read();
        if ((unsigned int)result.type == ED_TYPE_SPIKE) {
            if (spike_count == 0 && first_spike != 0)
                *first_spike = result;
            ++spike_count;
            continue;
        }
        assert((unsigned int)result.type == (unsigned int)expected_marker_type);
        if (end_marker != 0) *end_marker = result;
        assert(output.empty());
        return spike_count;
    }
}

int main() {
    {
        ed_voltage_t shifted = 8;
        assert(close((double)(shifted >> 2), 2.0));
        shifted = -8;
        assert(close((double)(shifted >> 2), -2.0));
        shifted <<= 1;
        assert(close((double)shifted, -16.0));
        shifted >>= 3;
        assert(close((double)shifted, -2.0));

        ap_uint<6> active_shifts[ED_ACTIVE_SHIFT_TERMS] = {1, 2, 0, 0};
        ap_uint<3> active_terms = 2;
        assert(close(
            (double)ed_active_shift_scale(
                (ed_accum_t)8, active_shifts, active_terms),
            6.0));
        assert(close(
            (double)ed_active_shift_scale(
                (ed_accum_t)-8, active_shifts, active_terms),
            -6.0));

        assert(close((double)ed_exp_pwl_lut_decay_fp((ed_accum_t)0), 1.0));
        assert(close((double)ed_exp_pwl_lut_decay_fp((ed_accum_t)-1), 0.3678794412));
        assert(close((double)ed_exp_pwl_lut_decay_fp((ed_accum_t)-9), 0.0));

        ed_voltage_t voltage = 2;
        ed_current_t current = 0;
        ed_current_t spike = ed_update_cuba_lif_neuron(
            &voltage, &current, (ed_time_step_t)0, (ed_current_t)0,
            (ed_weight_t)1, (ed_weight_t)1, (ed_tau_t)1, (ed_tau_t)1,
            (ed_voltage_t)1, (ed_voltage_t)0, (ed_voltage_t)0,
            ed_exp_pwl_lut_decay_fp, true
        );
        assert(close((double)spike, 1.0));
        assert(close((double)voltage, 1.0));

        ed_accum_t logarithm = 0;
        assert(ed_log_pwl_fp((ed_accum_t)1, &logarithm));
        assert(std::fabs((double)logarithm) < 1e-6);
        assert(ed_log_pwl_fp((ed_accum_t)2, &logarithm));
        assert(std::fabs((double)logarithm - std::log(2.0)) < 2e-4);
        assert(ed_log_pwl_fp((ed_accum_t)0.25, &logarithm));
        assert(std::fabs((double)logarithm - std::log(0.25)) < 2e-4);
        assert(!ed_log_pwl_fp((ed_accum_t)0, &logarithm));

        ed_time_step_t predicted = 0;
        assert(ed_predict_cuba_lif_spike<true>(
            (ed_voltage_t)0, (ed_current_t)1,
            (ed_accum_t)10, (ed_accum_t)0.001, (ed_accum_t)0.0002,
            (ed_accum_t)0.7, (ed_accum_t)0,
            &predicted));
        assert(std::fabs((double)predicted - 0.000091634405) < 0.00001);
    }

    {
        hls::stream<ed_spike_t> input;
        hls::stream<ed_spike_t> convolution;
        hls::stream<ed_spike_t> pooling;
        ed_weight_t weights[1][1][1][1] = {{{{2}}}};
        float time_driven_input[1][2][2] = {{{1, 2}, {3, 4}}};
        float time_driven_convolution[1][2][2] = {};
        float time_driven_pooling[1][1][1] = {};
        Conv2d<1, 1, 1, 1, 0, 0, 1, 1, 1>(
            time_driven_input, time_driven_convolution, weights);
        SumPool2d<2, 2, 2, 2, 0, 0>(time_driven_convolution, time_driven_pooling);

        input.write(event(0, 0, 0, 1));
        input.write(event(0, 0, 1, 2));
        input.write(event(0, 1, 0, 3));
        input.write(event(0, 1, 1, 4));
        input.write(marker(ED_TYPE_END_STEP, 3));

        Conv2d<1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1>(
            input, convolution, weights);
        SumPool2d<2, 2, 2, 2, 0, 0, 1, 2, 2>(convolution, pooling);

        ed_spike_t result = pooling.read();
        assert((unsigned int)result.type == ED_TYPE_SPIKE);
        assert(close((double)result.amplitude, time_driven_pooling[0][0][0]));
        ed_spike_t end = pooling.read();
        assert((unsigned int)end.type == ED_TYPE_END_STEP);
        assert((unsigned int)end.time_step == 3);
        assert((unsigned int)end.batch_idx == 7);
        assert(pooling.empty());
    }

    {
        hls::stream<ed_spike_t> input;
        hls::stream<ed_spike_t> output;
        ed_weight_t weights[1][1][1][1] = {{{{1}}}};
        ed_weight_t bias[1] = {0.5};
        input.write(marker(ED_TYPE_END_STEP, 1));
        Conv2d<1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1>(
            input, output, weights, bias);
        assert(close((double)output.read().amplitude, 0.5));
        assert((unsigned int)output.read().type == ED_TYPE_END_STEP);
    }

    {
        ed_weight_t r[1][1][1] = {{{1}}};
        ed_weight_t threshold[1][1][1] = {{{1.5}}};
        ed_weight_t reset[1][1][1] = {{{0}}};

        hls::stream<ed_spike_t> step_one_in;
        hls::stream<ed_spike_t> step_one_out;
        step_one_in.write(event(0, 0, 0, 1));
        step_one_in.write(marker(ED_TYPE_END_STEP, 1));
        IF<1, 1, 1>(step_one_in, step_one_out, r, threshold, reset);
        assert((unsigned int)step_one_out.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> step_two_in;
        hls::stream<ed_spike_t> step_two_out;
        step_two_in.write(event(0, 0, 0, 1));
        step_two_in.write(marker(ED_TYPE_END_SAMPLE, 2));
        IF<1, 1, 1>(step_two_in, step_two_out, r, threshold, reset);
        assert((unsigned int)step_two_out.read().type == ED_TYPE_SPIKE);
        assert((unsigned int)step_two_out.read().type == ED_TYPE_END_SAMPLE);
        assert(step_two_out.empty());
    }

    {
        ed_weight_t tau[1][1][1] = {{{2}}};
        ed_weight_t r[1][1][1] = {{{1}}};
        ed_weight_t leak[1][1][1] = {{{0}}};
        ed_weight_t threshold[1][1][1] = {{{0.7}}};
        ed_weight_t reset[1][1][1] = {{{0}}};

        hls::stream<ed_spike_t> first_in;
        hls::stream<ed_spike_t> first_out;
        first_in.write(event(0, 0, 0, 1));
        first_in.write(marker(ED_TYPE_END_STEP, 1));
        LIF<1, 1, 1>(first_in, first_out, tau, r, leak, threshold, reset, ed_weight_t(1));
        assert((unsigned int)first_out.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> second_in;
        hls::stream<ed_spike_t> second_out;
        second_in.write(event(0, 0, 0, 1));
        second_in.write(marker(ED_TYPE_END_SAMPLE, 2));
        LIF<1, 1, 1>(second_in, second_out, tau, r, leak, threshold, reset, ed_weight_t(1));
        assert((unsigned int)second_out.read().type == ED_TYPE_SPIKE);
        assert((unsigned int)second_out.read().type == ED_TYPE_END_SAMPLE);
    }

    {
        // Generic LIF evolves only when an input arrives.  The second event
        // sees the PWL decay from timestamp 0 to 2 and emits immediately at
        // timestamp 2, rather than waiting for the end-of-step marker.
        ed_weight_t tau[1][1][1] = {{{2}}};
        ed_weight_t r[1][1][1] = {{{1}}};
        ed_weight_t leak[1][1][1] = {{{0}}};
        ed_weight_t threshold[1][1][1] = {{{0.65}}};
        ed_weight_t reset[1][1][1] = {{{0}}};

        hls::stream<ed_spike_t> first_in, first_out;
        ed_spike_t first = event(0, 0, 0, 1);
        first.timestamp = 0;
        first_in.write(first);
        first_in.write(marker(ED_TYPE_END_STEP, 1));
        LIF<1, 1, 1, 314>(
            first_in, first_out, tau, r, leak, threshold, reset,
            ed_weight_t(1));
        assert((unsigned int)first_out.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> idle_in, idle_out;
        idle_in.write(marker(ED_TYPE_END_STEP, 10));
        LIF<1, 1, 1, 314>(
            idle_in, idle_out, tau, r, leak, threshold, reset,
            ed_weight_t(1));
        assert((unsigned int)idle_out.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> second_in, second_out;
        ed_spike_t second = event(0, 0, 0, 1);
        second.timestamp = 2;
        second_in.write(second);
        second_in.write(marker(ED_TYPE_END_SAMPLE, 3));
        LIF<1, 1, 1, 314>(
            second_in, second_out, tau, r, leak, threshold, reset,
            ed_weight_t(1));
        ed_spike_t emitted = second_out.read();
        assert((unsigned int)emitted.type == ED_TYPE_SPIKE);
        assert(close((double)emitted.timestamp, 2.0));
        assert((unsigned int)second_out.read().type == ED_TYPE_END_SAMPLE);
    }

    {
        // Compile and exercise the 1D LIF overload as well.
        ed_weight_t tau[1] = {2};
        ed_weight_t r[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        hls::stream<ed_spike_t> input, output;
        ed_spike_t spike = event(0, 0, 0, 2);
        spike.timestamp = 0;
        input.write(spike);
        input.write(marker(ED_TYPE_END_SAMPLE, 1));
        LIF<1, 315>(input, output, tau, r, leak, threshold, reset,
                     ed_weight_t(1));
        assert((unsigned int)output.read().type == ED_TYPE_SPIKE);
        assert((unsigned int)output.read().type == ED_TYPE_END_SAMPLE);
    }

    {
        hls::stream<ed_spike_t> split_in, split_one, split_two;
        split_in.write(marker(ED_TYPE_END_STEP, 4));
        Split(split_in, split_one, split_two);
        assert((unsigned int)split_one.read().type == ED_TYPE_END_STEP);
        assert((unsigned int)split_two.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> merge_one, merge_two, merged;
        merge_one.write(marker(ED_TYPE_END_SAMPLE, 5));
        merge_two.write(marker(ED_TYPE_END_SAMPLE, 5));
        Merge(merge_one, merge_two, merged);
        assert((unsigned int)merged.read().type == ED_TYPE_END_SAMPLE);

        hls::stream<ed_spike_t> flatten_in, flatten_out;
        flatten_in.write(marker(ED_TYPE_END_STEP, 6));
        Flatten<1, 2, 2>(flatten_in, flatten_out);
        assert((unsigned int)flatten_out.read().type == ED_TYPE_END_STEP);
    }

    {
        ed_weight_t weights[1][1] = {{1}};
        ed_weight_t bias[1] = {0};
        hls::stream<ed_spike_t> linear_in, linear_out;
        linear_in.write(marker(ED_TYPE_END_STEP, 7));
        Linear<1, 1>(linear_in, linear_out, weights);
        assert((unsigned int)linear_out.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> affine_in, affine_out;
        affine_in.write(marker(ED_TYPE_END_SAMPLE, 8));
        Affine<1, 1>(affine_in, affine_out, weights, bias);
        assert((unsigned int)affine_out.read().type == ED_TYPE_END_SAMPLE);
    }

    {
        ed_tau_t tau_syn[1] = {2};
        ed_tau_t tau_mem[1] = {3};
        ed_weight_t r[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {10};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_weight[1] = {1};
        hls::stream<ed_spike_t> cuba_in, cuba_out;
        cuba_in.write(marker(ED_TYPE_END_SAMPLE, 9));
        CubaLIF<1, 1, 1>(cuba_in, cuba_out, tau_syn, tau_mem, r, leak,
                         threshold, reset, input_weight);
        assert((unsigned int)cuba_out.read().type == ED_TYPE_END_SAMPLE);
    }

    {
        ed_tau_t tau_syn[1] = {0.0002};
        ed_tau_t tau_mem[1] = {0.001};
        ed_weight_t r[1] = {10};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_weight[1] = {1};

        hls::stream<ed_spike_t> input, output;
        ed_spike_t excitation = event(0, 0, 0, 1);
        excitation.timestamp = 0;
        input.write(excitation);
        ed_spike_t end = marker(ED_TYPE_END_STEP, 0);
        end.timestamp = 0.0002;
        input.write(end);
        CubaLIF<1, 1, 1, 101, true, false>(
            input, output, tau_syn, tau_mem, r, leak, threshold, reset,
            input_weight, true);
        ed_spike_t self_spike = output.read();
        assert((unsigned int)self_spike.type == ED_TYPE_SPIKE);
        assert((double)self_spike.timestamp > 0.00008);
        assert((double)self_spike.timestamp < 0.00011);
        assert((unsigned int)output.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> cancel_input, cancel_output;
        excitation.timestamp = 0;
        cancel_input.write(excitation);
        ed_spike_t inhibition = event(0, 0, 0, -0.8);
        inhibition.timestamp = 0.00005;
        cancel_input.write(inhibition);
        end.timestamp = 0.0002;
        cancel_input.write(end);
        CubaLIF<1, 1, 1, 102, true, false>(
            cancel_input, cancel_output, tau_syn, tau_mem, r, leak,
            threshold, reset, input_weight, true);
        assert((unsigned int)cancel_output.read().type == ED_TYPE_END_STEP);

        hls::stream<ed_spike_t> reschedule_input, reschedule_output;
        excitation.timestamp = 0;
        reschedule_input.write(excitation);
        ed_spike_t reinforcement = event(0, 0, 0, 1);
        reinforcement.timestamp = 0.00005;
        reschedule_input.write(reinforcement);
        end.timestamp = 0.0002;
        reschedule_input.write(end);
        CubaLIF<1, 1, 1, 105, true, false>(
            reschedule_input, reschedule_output, tau_syn, tau_mem, r, leak,
            threshold, reset, input_weight, true);
        ed_spike_t rescheduled = reschedule_output.read();
        assert((unsigned int)rescheduled.type == ED_TYPE_SPIKE);
        assert((double)rescheduled.timestamp > 0.00005);
        assert((double)rescheduled.timestamp < 0.000091);
        assert((unsigned int)reschedule_output.read().type == ED_TYPE_END_STEP);
    }

    {
        ed_tau_t tau_syn[1] = {0.0002};
        ed_tau_t tau_mem[1] = {0.001};
        ed_weight_t r[1] = {10};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_weight[1] = {1};
        ed_spike_t excitation = event(0, 0, 0, 3);
        excitation.timestamp = 0;
        ed_spike_t end = marker(ED_TYPE_END_SAMPLE, 0);
        end.timestamp = 0.0002;

        hls::stream<ed_spike_t> continuous_input, continuous_output;
        continuous_input.write(excitation);
        continuous_input.write(end);
        CubaLIF<1, 1, 1, 103, true, false, true>(
            continuous_input, continuous_output, tau_syn, tau_mem, r, leak,
            threshold, reset, input_weight, true);
        int continuous_spikes = 0;
        double previous_time = -1;
        while (true) {
            ed_spike_t result = continuous_output.read();
            if (result.type != ED_TYPE_SPIKE) break;
            assert((double)result.timestamp > previous_time);
            previous_time = (double)result.timestamp;
            ++continuous_spikes;
        }
        assert(continuous_spikes > 1);

        hls::stream<ed_spike_t> discrete_input, discrete_output;
        discrete_input.write(excitation);
        discrete_input.write(end);
        CubaLIF<1, 1, 1, 104, true, false, false>(
            discrete_input, discrete_output, tau_syn, tau_mem, r, leak,
            threshold, reset, input_weight, true);
        int discrete_spikes = 0;
        while (true) {
            ed_spike_t result = discrete_output.read();
            if (result.type != ED_TYPE_SPIKE) break;
            ++discrete_spikes;
        }
        assert(discrete_spikes == 1);
    }

    {
        // One event activates the neuron.  The membrane crosses the threshold
        // only on the following lightweight tick, with no new input event.
        ap_uint<6> decay_u_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{2, 0, 0, 0}};
        ap_uint<3> decay_u_terms[1] = {1};
        ap_uint<6> decay_v_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{2, 0, 0, 0}};
        ap_uint<3> decay_v_terms[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_gain[1] = {0.5};

        hls::stream<ed_spike_t> first_input, first_output;
        first_input.write(event(0, 0, 0, 1));
        first_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 201, false>(
            first_input, first_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(first_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> second_input, second_output;
        second_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 201, false>(
            second_input, second_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        ed_spike_t delayed_spike = {};
        ed_spike_t delayed_marker = {};
        assert(drain_window(
            second_output, ED_TYPE_END_STEP,
            &delayed_spike, &delayed_marker) == 1);
        assert(close((double)delayed_spike.timestamp, 2.0));
        assert((unsigned int)delayed_spike.time_step == 2);
        assert((unsigned int)delayed_spike.batch_idx == 7);
        assert((unsigned int)delayed_spike.channel_idx == 0);
        assert((unsigned int)delayed_spike.height_idx == 0);
        assert((unsigned int)delayed_spike.width_idx == 0);
        assert((unsigned int)delayed_marker.time_step == 2);
    }

    {
        // An inhibitory event arriving before the delayed crossing cancels it.
        // Multiple excitatory events for one ID are accumulated but the ID is
        // processed only once by the active-list tick.
        ap_uint<6> decay_u_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{2, 0, 0, 0}};
        ap_uint<3> decay_u_terms[1] = {1};
        ap_uint<6> decay_v_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{2, 0, 0, 0}};
        ap_uint<3> decay_v_terms[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_gain[1] = {0.5};

        hls::stream<ed_spike_t> excite_input, excite_output;
        excite_input.write(event(0, 0, 0, 1));
        excite_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 202, false>(
            excite_input, excite_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(excite_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> inhibit_input, inhibit_output;
        inhibit_input.write(event(0, 0, 0, -1));
        inhibit_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 202, false>(
            inhibit_input, inhibit_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(inhibit_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> silent_input, silent_output;
        silent_input.write(marker(ED_TYPE_END_STEP, 3));
        CubaLIFActiveList<1, 1, 1, 202, false>(
            silent_input, silent_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(silent_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> aggregate_input, aggregate_output;
        aggregate_input.write(event(0, 0, 0, 0.75));
        aggregate_input.write(event(0, 0, 0, 0.75));
        aggregate_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 203, true>(
            aggregate_input, aggregate_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(aggregate_output, ED_TYPE_END_STEP) == 1);

        // If the same neuron had been inserted twice, its state would have
        // advanced twice above and this next tick would fire spuriously.
        hls::stream<ed_spike_t> aggregate_next_input, aggregate_next_output;
        aggregate_next_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 203, true>(
            aggregate_next_input, aggregate_next_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(aggregate_next_output, ED_TYPE_END_STEP) == 0);

        // A negative event must also activate an idle neuron.  Otherwise the
        // following excitation would incorrectly ignore the inhibitory state
        // and cross the threshold immediately.
        hls::stream<ed_spike_t> negative_input, negative_output;
        negative_input.write(event(0, 0, 0, -1));
        negative_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 207, false>(
            negative_input, negative_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(negative_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> after_negative_input, after_negative_output;
        after_negative_input.write(event(0, 0, 0, 1.5));
        after_negative_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 207, false>(
            after_negative_input, after_negative_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(after_negative_output, ED_TYPE_END_STEP) == 0);
    }

    {
        // Once both states fall below the noise floor the ID is removed.  A
        // later event must clear the membership bit, reinsert it and fire.
        ap_uint<6> decay_u_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{1, 0, 0, 0}};
        ap_uint<3> decay_u_terms[1] = {1};
        ap_uint<6> decay_v_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{1, 0, 0, 0}};
        ap_uint<3> decay_v_terms[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {2};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_gain[1] = {0.3};

        hls::stream<ed_spike_t> activate_input, activate_output;
        activate_input.write(event(0, 0, 0, 1));
        activate_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 204, false>(
            activate_input, activate_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.2, true);
        assert(drain_window(activate_output, ED_TYPE_END_STEP) == 0);

        for (int step = 2; step <= 4; ++step) {
            hls::stream<ed_spike_t> decay_input, decay_output;
            decay_input.write(marker(ED_TYPE_END_STEP, step));
            CubaLIFActiveList<1, 1, 1, 204, false>(
                decay_input, decay_output,
                decay_u_shifts, decay_u_terms,
                decay_v_shifts, decay_v_terms,
                leak, threshold, reset, input_gain,
                (ed_accum_t)0.2, false);
            assert(drain_window(decay_output, ED_TYPE_END_STEP) == 0);
        }

        hls::stream<ed_spike_t> reinsert_input, reinsert_output;
        reinsert_input.write(event(0, 0, 0, 10));
        reinsert_input.write(marker(ED_TYPE_END_STEP, 5));
        CubaLIFActiveList<1, 1, 1, 204, false>(
            reinsert_input, reinsert_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.2, false);
        assert(drain_window(reinsert_output, ED_TYPE_END_STEP) == 1);
    }

    {
        // Explicit reset clears a pending delayed spike and all membership.
        // The same neuron can subsequently be activated again.
        ap_uint<6> decay_u_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{2, 0, 0, 0}};
        ap_uint<3> decay_u_terms[1] = {1};
        ap_uint<6> decay_v_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{2, 0, 0, 0}};
        ap_uint<3> decay_v_terms[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_gain[1] = {0.5};

        hls::stream<ed_spike_t> before_reset_input, before_reset_output;
        before_reset_input.write(event(0, 0, 0, 1));
        before_reset_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 205, false>(
            before_reset_input, before_reset_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(before_reset_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> reset_input, reset_output;
        reset_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 205, false>(
            reset_input, reset_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(reset_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> after_reset_input, after_reset_output;
        after_reset_input.write(marker(ED_TYPE_END_STEP, 3));
        CubaLIFActiveList<1, 1, 1, 205, false>(
            after_reset_input, after_reset_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(after_reset_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> reactivate_input, reactivate_output;
        reactivate_input.write(event(0, 0, 0, 1));
        reactivate_input.write(marker(ED_TYPE_END_STEP, 4));
        CubaLIFActiveList<1, 1, 1, 205, false>(
            reactivate_input, reactivate_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(reactivate_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> delayed_input, delayed_output;
        delayed_input.write(marker(ED_TYPE_END_STEP, 5));
        CubaLIFActiveList<1, 1, 1, 205, false>(
            delayed_input, delayed_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(delayed_output, ED_TYPE_END_STEP) == 1);

        hls::stream<ed_spike_t> sample_input, sample_output;
        sample_input.write(event(0, 0, 0, 1));
        sample_input.write(marker(ED_TYPE_END_SAMPLE, 6));
        CubaLIFActiveList<1, 1, 1, 206, false>(
            sample_input, sample_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, true);
        assert(drain_window(sample_output, ED_TYPE_END_SAMPLE) == 0);

        hls::stream<ed_spike_t> next_sample_input, next_sample_output;
        next_sample_input.write(marker(ED_TYPE_END_STEP, 7));
        CubaLIFActiveList<1, 1, 1, 206, false>(
            next_sample_input, next_sample_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.001, false);
        assert(drain_window(next_sample_output, ED_TYPE_END_STEP) == 0);
    }

    {
        // A sub-epsilon event is still consumed by its own watermark.  It
        // must not remain pending across a silent step and combine with a
        // later event as though both had arrived together.
        ap_uint<6> decay_u_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{1, 0, 0, 0}};
        ap_uint<3> decay_u_terms[1] = {1};
        ap_uint<6> decay_v_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{1, 0, 0, 0}};
        ap_uint<3> decay_v_terms[1] = {1};
        ed_weight_t leak[1] = {0};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_gain[1] = {0.4};

        hls::stream<ed_spike_t> first_input, first_output;
        first_input.write(event(0, 0, 0, 1));
        first_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 208, false>(
            first_input, first_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.5, true);
        assert(drain_window(first_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> silent_input, silent_output;
        silent_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 208, false>(
            silent_input, silent_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.5, false);
        assert(drain_window(silent_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> later_input, later_output;
        later_input.write(event(0, 0, 0, 1));
        later_input.write(marker(ED_TYPE_END_STEP, 3));
        CubaLIFActiveList<1, 1, 1, 208, false>(
            later_input, later_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.5, false);
        assert(drain_window(later_output, ED_TYPE_END_STEP) == 0);
    }

    {
        // If reset and leak differ, the membrane is active even without
        // synaptic current and must relax toward leak on subsequent ticks.
        ap_uint<6> decay_u_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{1, 0, 0, 0}};
        ap_uint<3> decay_u_terms[1] = {1};
        ap_uint<6> decay_v_shifts[1][ED_ACTIVE_SHIFT_TERMS] = {{1, 0, 0, 0}};
        ap_uint<3> decay_v_terms[1] = {1};
        ed_weight_t leak[1] = {1};
        ed_weight_t threshold[1] = {0.7};
        ed_weight_t reset[1] = {0};
        ed_weight_t input_gain[1] = {0};

        hls::stream<ed_spike_t> first_input, first_output;
        first_input.write(marker(ED_TYPE_END_STEP, 1));
        CubaLIFActiveList<1, 1, 1, 209, false>(
            first_input, first_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.01, true);
        assert(drain_window(first_output, ED_TYPE_END_STEP) == 0);

        hls::stream<ed_spike_t> second_input, second_output;
        second_input.write(marker(ED_TYPE_END_STEP, 2));
        CubaLIFActiveList<1, 1, 1, 209, false>(
            second_input, second_output,
            decay_u_shifts, decay_u_terms,
            decay_v_shifts, decay_v_terms,
            leak, threshold, reset, input_gain,
            (ed_accum_t)0.01, false);
        assert(drain_window(second_output, ED_TYPE_END_STEP) == 1);
    }

    return 0;
}
