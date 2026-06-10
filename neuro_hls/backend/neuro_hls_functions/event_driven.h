#pragma once

#include <ap_fixed.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdbool.h>

// =============================================================
// TIPOS — baseado em event_driven_srnn/types.h
// =============================================================

typedef ap_fixed<40, 24> ed_voltage_t;
typedef ap_fixed<16, 8>  ed_current_t;
typedef ap_fixed<24, 8>  ed_time_step_t;
typedef ap_fixed<24, 8>  ed_tau_t;
typedef ap_fixed<24, 8>  ed_decay_t;
typedef ap_fixed<16, 8>  ed_weight_t;
typedef ap_fixed<40, 24> ed_accum_t;

typedef struct {
    ap_uint<4>   type;
    ed_current_t amplitude;
    ed_time_step_t timestamp;
    ap_uint<32>  time_step;
    ap_uint<32>  batch_idx;
    ap_uint<32>  channel_idx;
    ap_uint<32>  height_idx;
    ap_uint<32>  width_idx;
} ed_spike_t;

static const int ED_TYPE_SPIKE      = 0;
static const int ED_TYPE_END_STEP   = 1;
static const int ED_TYPE_END_SAMPLE = 2;

// =============================================================
// TS-EFA — baseado em event_driven_srnn/ts_efa.h
// =============================================================

#define ED_FRACTIONAL_BITS  8
#define ED_ONE_FP           (1 << ED_FRACTIONAL_BITS)
#define ED_DECAY_FRAC_BITS  16
#define ED_DECAY_ONE_FP     (1u << ED_DECAY_FRAC_BITS)
#define ED_LUT_SIZE         256

static const uint32_t ED_TEMP_LUT[ED_LUT_SIZE] = {
    65536, 65359, 65182, 65006, 64830, 64655, 64480, 64306,
    64132, 63958, 63785, 63613, 63441, 63269, 63098, 62928,
    62757, 62588, 62419, 62250, 62081, 61914, 61746, 61579,
    61413, 61247, 61081, 60916, 60751, 60587, 60423, 60260,
    60097, 59934, 59772, 59611, 59449, 59289, 59128, 58968,
    58809, 58650, 58491, 58333, 58176, 58018, 57861, 57705,
    57549, 57393, 57238, 57083, 56929, 56775, 56622, 56468,
    56316, 56163, 56012, 55860, 55709, 55558, 55408, 55258,
    55109, 54960, 54811, 54663, 54515, 54368, 54221, 54074,
    53928, 53782, 53637, 53492, 53347, 53203, 53059, 52916,
    52773, 52630, 52488, 52346, 52204, 52063, 51922, 51782,
    51642, 51502, 51363, 51224, 51085, 50947, 50810, 50672,
    50535, 50399, 50262, 50126, 49991, 49856, 49721, 49586,
    49452, 49319, 49185, 49052, 48920, 48787, 48655, 48524,
    48393, 48262, 48131, 48001, 47871, 47742, 47613, 47484,
    47356, 47228, 47100, 46973, 46846, 46719, 46593, 46467,
    46341, 46216, 46091, 45966, 45842, 45718, 45594, 45471,
    45348, 45225, 45103, 44981, 44859, 44738, 44617, 44497,
    44376, 44256, 44137, 44017, 43898, 43780, 43661, 43543,
    43425, 43308, 43191, 43074, 42958, 42841, 42726, 42610,
    42495, 42380, 42265, 42151, 42037, 41923, 41810, 41697,
    41584, 41472, 41360, 41248, 41136, 41025, 40914, 40804,
    40693, 40583, 40473, 40364, 40255, 40146, 40037, 39929,
    39821, 39714, 39606, 39499, 39392, 39286, 39180, 39074,
    38968, 38863, 38757, 38653, 38548, 38444, 38340, 38236,
    38133, 38030, 37927, 37824, 37722, 37620, 37518, 37417,
    37316, 37215, 37114, 37014, 36914, 36814, 36715, 36615,
    36516, 36417, 36319, 36221, 36123, 36025, 35928, 35831,
    35734, 35637, 35541, 35445, 35349, 35253, 35158, 35063,
    34968, 34874, 34779, 34685, 34591, 34498, 34405, 34312,
    34219, 34126, 34034, 33942, 33850, 33759, 33667, 33576,
    33486, 33395, 33305, 33215, 33125, 33035, 32946, 32857
};

static const uint32_t ED_SCAL_LUT[ED_LUT_SIZE] = {
    65536, 32768, 16384, 8192, 4096, 2048, 1024, 512,
    256,   128,   64,    32,   16,   8,    4,    2,
    1,     1
};

inline uint32_t ed_ts_efa_compute_decay(uint32_t dt_scaled) {
    #pragma HLS INLINE off
    uint32_t dt_int  = dt_scaled >> ED_FRACTIONAL_BITS;
    uint32_t dt_frac = dt_scaled & (ED_ONE_FP - 1);
    if (dt_int >= ED_LUT_SIZE) return 0;
    uint64_t product = (uint64_t)ED_TEMP_LUT[dt_frac] * (uint64_t)ED_SCAL_LUT[dt_int];
    return (uint32_t)((product + (ED_DECAY_ONE_FP >> 1)) >> ED_DECAY_FRAC_BITS);
}

inline ed_decay_t ed_ts_efa_compute_decay_fp(ed_accum_t exp_arg) {
    #pragma HLS INLINE off
    const ed_accum_t INV_LN2 = (ed_accum_t)1.44269504088896340736;
    if (exp_arg >= (ed_accum_t)0) return (ed_decay_t)1;
    ed_accum_t scaled = (-exp_arg) * INV_LN2;
    uint32_t scaled_fp = (uint32_t)(scaled * (ed_accum_t)ED_ONE_FP);
    uint32_t decay_fp  = ed_ts_efa_compute_decay(scaled_fp);
    ap_ufixed<17, 1> decay_value = 0;
    decay_value.range(16, 0) = (ap_uint<17>)decay_fp;
    return (ed_decay_t)decay_value;
}

// =============================================================
// CUBA-LIF KERNEL — baseado em event_driven_srnn/cuba_lif.h
// =============================================================

typedef ed_decay_t (*ed_exp_decay_func_ptr)(ed_accum_t exp_arg);

inline ed_current_t ed_update_cuba_lif_neuron(
    ed_voltage_t*       v_mem,
    ed_current_t*       i_syn,
    ed_time_step_t      dt,
    ed_current_t        input_current,
    ed_weight_t         input_weight,
    ed_weight_t         r,
    ed_tau_t            tau_mem,
    ed_tau_t            tau_syn,
    ed_voltage_t        v_th,
    ed_voltage_t        v_leak,
    ed_voltage_t        v_reset,
    ed_exp_decay_func_ptr exp_decay_func
) {
    #pragma HLS INLINE off
    const ed_accum_t dt_f   = (ed_accum_t)dt;
    const ed_accum_t tau_m  = (ed_accum_t)tau_mem;
    const ed_accum_t tau_s  = (ed_accum_t)tau_syn;
    const ed_accum_t v0     = (ed_accum_t)(*v_mem);
    const ed_accum_t i0     = (ed_accum_t)(*i_syn);

    const ed_accum_t decay_mem = (ed_accum_t)exp_decay_func(-(dt_f / tau_m));
    const ed_accum_t decay_syn = (ed_accum_t)exp_decay_func(-(dt_f / tau_s));

    ed_accum_t v_t = (ed_accum_t)v_leak + (v0 - (ed_accum_t)v_leak) * decay_mem;
    const ed_accum_t tau_diff = tau_m - tau_s;
    if (tau_diff != (ed_accum_t)0) {
        v_t += ((ed_accum_t)r * i0 * tau_s / tau_diff) * (decay_mem - decay_syn);
    } else {
        v_t += ((ed_accum_t)r * i0 / tau_m) * dt_f * decay_mem;
    }

    ed_accum_t i_t     = i0 * decay_syn;
    ed_accum_t delta_i = (ed_accum_t)input_weight * (ed_accum_t)input_current;
    *i_syn = (ed_current_t)(i_t + delta_i);

    if (v_t >= (ed_accum_t)v_th) {
        *v_mem = v_reset;
        return (ed_current_t)1;
    }
    *v_mem = (ed_voltage_t)v_t;
    return (ed_current_t)0;
}

// =============================================================
// SPLIT — baseado em event_driven_srnn/split.h
// =============================================================

inline void Split(
    hls::stream<ed_spike_t>& in,
    hls::stream<ed_spike_t>& out1,
    hls::stream<ed_spike_t>& out2)
{
    bool step_done = false;
    while (!step_done) {
        if (!in.empty()) {
            ed_spike_t s = in.read();
            out1.write(s);
            out2.write(s);
            if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE)
                step_done = true;
        }
    }
}

// =============================================================
// MERGE — baseado em event_driven_srnn/merge.h
// =============================================================

inline void Merge(
    hls::stream<ed_spike_t>& in1,
    hls::stream<ed_spike_t>& in2,
    hls::stream<ed_spike_t>& out)
{
    bool in1_done = false;
    bool in2_done = false;
    ed_spike_t end_token;

    while (!in1_done) {
        ed_spike_t s = in1.read();
        if (s.type == ED_TYPE_SPIKE) {
            out.write(s);
        } else {
            end_token = s;
            in1_done  = true;
        }
    }

    while (!in2_done) {
        ed_spike_t s = in2.read();
        if (s.type == ED_TYPE_SPIKE) {
            s.timestamp  = end_token.timestamp;
            s.time_step  = end_token.time_step;
            out.write(s);
        } else {
            in2_done = true;
        }
    }

    out.write(end_token);
}

// =============================================================
// LINEAR — baseado em event_driven_srnn/linear.h
// =============================================================

template<int NUM_INPUTS, int NUM_OUTPUTS>
void Linear(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const ed_weight_t weights[NUM_OUTPUTS][NUM_INPUTS])
{
    static ed_accum_t acc_buffer[NUM_OUTPUTS];
    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            #pragma HLS PIPELINE II=1
            acc_buffer[i] = 0;
        }
        initialized = true;
    }

    bool end_of_processing = false;
    main_loop: while (!end_of_processing) {
        ed_spike_t s = input_stream.read();

        if (s.type == ED_TYPE_SPIKE) {
            unsigned int in_idx = s.width_idx;
            ed_current_t amp = s.amplitude;
            if (in_idx < NUM_INPUTS) {
                update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                    #pragma HLS PIPELINE
                    acc_buffer[out_idx] += weights[out_idx][in_idx] * amp;
                }
            }
        } else if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE) {
            output_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS PIPELINE
                ed_spike_t out_spike;
                out_spike.type        = ED_TYPE_SPIKE;
                out_spike.amplitude   = (ed_current_t)acc_buffer[out_idx];
                out_spike.channel_idx = 0;
                out_spike.height_idx  = 0;
                out_spike.width_idx   = out_idx;
                out_spike.timestamp   = s.timestamp;
                out_spike.time_step   = s.time_step;
                out_spike.batch_idx   = s.batch_idx;
                output_stream.write(out_spike);
                acc_buffer[out_idx] = 0;
            }
            output_stream.write(s);
            end_of_processing = true;
        }
    }
}

// =============================================================
// AFFINE — baseado em event_driven_srnn/affine.h
// =============================================================

template<int NUM_INPUTS, int NUM_OUTPUTS>
void Affine(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    ed_weight_t weights[NUM_OUTPUTS][NUM_INPUTS],
    ed_weight_t biases[NUM_OUTPUTS])
{
    static ed_accum_t acc_buffer[NUM_OUTPUTS];
    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            #pragma HLS PIPELINE II=1
            acc_buffer[i] = biases[i];
        }
        initialized = true;
    }

    bool end_of_processing = false;
    main_loop: while (!input_stream.empty()) {
        ed_spike_t s = input_stream.read();

        if (s.type == ED_TYPE_SPIKE) {
            unsigned int in_idx  = s.width_idx;
            ed_current_t amp     = s.amplitude;
            if (in_idx < NUM_INPUTS) {
                update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                    #pragma HLS PIPELINE
                    acc_buffer[out_idx] += (ed_accum_t)weights[out_idx][in_idx] * (ed_accum_t)amp;
                }
            }
        } else if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE) {
            output_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; out_idx++) {
                #pragma HLS PIPELINE
                ed_spike_t out_spike;
                out_spike.type        = ED_TYPE_SPIKE;
                out_spike.amplitude   = (ed_current_t)acc_buffer[out_idx];
                out_spike.channel_idx = 0;
                out_spike.height_idx  = 0;
                out_spike.width_idx   = out_idx;
                out_spike.timestamp   = s.timestamp;
                out_spike.time_step   = s.time_step;
                out_spike.batch_idx   = s.batch_idx;
                output_stream.write(out_spike);
                acc_buffer[out_idx] = biases[out_idx]; // Reinicia com bias para o próximo passo
            }
            output_stream.write(s);
            end_of_processing = true;
        }
    }
}

// =============================================================
// CUBA-LIF WRAPPER — baseado em event_driven_srnn/cuba_lif.h
// =============================================================

template<int IN_C, int IN_H, int IN_W>
void CubaLIF(
    hls::stream<ed_spike_t>& in,
    hls::stream<ed_spike_t>& out,
    ed_tau_t    tau_syn[IN_W],
    ed_tau_t    tau_mem[IN_W],
    ed_weight_t r[IN_W],
    ed_weight_t v_leak[IN_W],
    ed_weight_t v_threshold[IN_W],
    ed_weight_t v_reset[IN_W],
    ed_weight_t w_in[IN_W])
{
    static ed_voltage_t  v_mem[IN_C][IN_H][IN_W] = {};
    static ed_current_t  i_syn[IN_C][IN_H][IN_W] = {};
    static ed_current_t  i_new[IN_C][IN_H][IN_W] = {};
    static ed_time_step_t last_time;
    static ed_time_step_t current_time;

    while (!in.empty()) {
        ed_spike_t s = in.read();

        if (s.type == ED_TYPE_SPIKE) {
            uint16_t c = s.channel_idx;
            uint16_t h = s.height_idx;
            uint16_t w = s.width_idx;
            i_new[c][h][w] += s.amplitude;

        } else if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE) {
            current_time = s.timestamp;

            for (int c = 0; c < IN_C; c++) {
                for (int h = 0; h < IN_H; h++) {
                    for (int w = 0; w < IN_W; w++) {
                        ed_time_step_t dt = current_time - last_time;

                        ed_current_t spike_amplitude = ed_update_cuba_lif_neuron(
                            &v_mem[c][h][w],
                            &i_syn[c][h][w],
                            dt,
                            i_new[c][h][w],
                            w_in[w],
                            r[w],
                            tau_mem[w],
                            tau_syn[w],
                            v_threshold[w],
                            v_leak[w],
                            v_reset[w],
                            ed_ts_efa_compute_decay_fp
                        );

                        if (spike_amplitude != (ed_current_t)0) {
                            ed_spike_t out_spk;
                            out_spk.type        = ED_TYPE_SPIKE;
                            out_spk.amplitude   = spike_amplitude;
                            out_spk.timestamp   = current_time;
                            out_spk.time_step   = s.time_step;
                            out_spk.batch_idx   = s.batch_idx;
                            out_spk.channel_idx = c;
                            out_spk.height_idx  = h;
                            out_spk.width_idx   = w;
                            out.write(out_spk);
                        }

                        i_new[c][h][w] = 0;
                    }
                }
            }

            last_time = current_time;
            out.write(s);

            if (s.type == ED_TYPE_END_SAMPLE) {
                last_time = 0;
                for (int c = 0; c < IN_C; c++)
                    for (int h = 0; h < IN_H; h++)
                        for (int w = 0; w < IN_W; w++) {
                            v_mem[c][h][w] = v_reset[w];
                            i_syn[c][h][w] = 0;
                        }
            }
            break;
        }
    }
}

// =============================================================
// FLATTEN — converte coordenadas 3D em índice linear no spike
// =============================================================

template<int CHANNELS, int HEIGHT, int WIDTH>
void Flatten(
    hls::stream<ed_spike_t>& in,
    hls::stream<ed_spike_t>& out)
{
    bool end_of_processing = false;
    while (!end_of_processing) {
        ed_spike_t s = in.read();
        if (s.type == ED_TYPE_SPIKE) {
            // Converte (channel, height, width) -> índice linear
            s.width_idx   = s.channel_idx * (HEIGHT * WIDTH) + s.height_idx * WIDTH + s.width_idx;
            s.channel_idx = 0;
            s.height_idx  = 0;
        }
        out.write(s);
        if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE)
            end_of_processing = true;
    }
}

// =============================================================
// STUBS — TODO: implementar Conv2d e SumPool2d event-driven
// =============================================================

// TODO: Conv2d event-driven
// template<...> void Conv2d(...) { /* a implementar */ }

// TODO: SumPool2d event-driven
// template<...> void SumPool2d(...) { /* a implementar */ }
