#pragma once

#include <ap_fixed.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdbool.h>

// =============================================================
// Scalar event-driven runtime types.  Actor-internal replication was retired.
// =============================================================

typedef ap_fixed<48, 24> ed_voltage_t;
typedef ap_fixed<32, 12> ed_current_t;
// Self-events need a substantially finer time grid than the external
// logical step.  With 24 fractional bits the internal tick is about 59.6 ns
// when timestamps are expressed in seconds.
typedef ap_fixed<32, 8>  ed_time_step_t;
typedef ap_fixed<32, 8>  ed_tau_t;
typedef ap_fixed<32, 8>  ed_decay_t;
typedef ap_fixed<16, 8>  ed_weight_t;
typedef ap_fixed<56, 24> ed_accum_t;

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

#define ED_INTERNAL_TIME_FRAC_BITS 24
#define ED_LOG_PWL_SEGMENTS 32
#define ED_ROOT_BISECTION_ITERATIONS 18
#define ED_MAX_SELF_EVENTS_PER_NEURON_PER_WINDOW 64
#define ED_ACTIVE_SHIFT_TERMS 4

static const ed_accum_t ED_INTERNAL_TIME_QUANTUM =
    (ed_accum_t)(1.0 / 16777216.0);

// The event-driven analytical backend uses the PWL decay approximation only.

// =============================================================
// PIECEWISE-LINEAR LUT — baseado em event_driven_srnn/exp_pwl_lut.h
// =============================================================

#define ED_EXP_PWL_LUT_SEGMENTS 64

static const ed_accum_t ED_EXP_PWL_LUT_X_MIN = (ed_accum_t)-8.0;
static const ed_accum_t ED_EXP_PWL_LUT_X_MAX = (ed_accum_t)0.0;
static const ed_accum_t ED_EXP_PWL_LUT_STEP = (ed_accum_t)0.125;
static const ed_accum_t ED_EXP_PWL_LUT_INV_STEP = (ed_accum_t)8.0;

static const ed_decay_t ED_EXP_PWL_LUT_VALUE[ED_EXP_PWL_LUT_SEGMENTS] = {
    0.0003354626, 0.0003801290, 0.0004307425, 0.0004880952,
    0.0005530844, 0.0006267267, 0.0007101744, 0.0008047330,
    0.0009118820, 0.0010332976, 0.0011708796, 0.0013267804,
    0.0015034392, 0.0017036198, 0.0019304541, 0.0021874911,
    0.0024787522, 0.0028087942, 0.0031827808, 0.0036065631,
    0.0040867714, 0.0046309187, 0.0052475184, 0.0059462174,
    0.0067379470, 0.0076350942, 0.0086516952, 0.0098036550,
    0.0111089965, 0.0125881422, 0.0142642339, 0.0161634946,
    0.0183156389, 0.0207543379, 0.0235177459, 0.0266490973,
    0.0301973834, 0.0342181183, 0.0387742078, 0.0439369336,
    0.0497870684, 0.0564161395, 0.0639278612, 0.0724397570,
    0.0820849986, 0.0930144892, 0.1053992246, 0.1194329683,
    0.1353352832, 0.1533549668, 0.1737739435, 0.1969116752,
    0.2231301601, 0.2528395958, 0.2865047969, 0.3246524674,
    0.3678794412, 0.4168620197, 0.4723665527, 0.5352614285,
    0.6065306597, 0.6872892788, 0.7788007831, 0.8824969026
};

static const ed_accum_t ED_EXP_PWL_LUT_SLOPE[ED_EXP_PWL_LUT_SEGMENTS] = {
    0.0003573306, 0.0004049087, 0.0004588216, 0.0005199130,
    0.0005891386, 0.0006675815, 0.0007564690, 0.0008571916,
    0.0009713254, 0.0011006559, 0.0012472065, 0.0014132701,
    0.0016014448, 0.0018146747, 0.0020562959, 0.0023300885,
    0.0026403361, 0.0029918928, 0.0033902587, 0.0038416664,
    0.0043531784, 0.0049327973, 0.0055895917, 0.0063338371,
    0.0071771778, 0.0081328079, 0.0092156787, 0.0104427320,
    0.0118331656, 0.0134087333, 0.0151940854, 0.0172171544,
    0.0195095919, 0.0221072639, 0.0250508118, 0.0283862887,
    0.0321658791, 0.0364487162, 0.0413018063, 0.0468010780,
    0.0530325691, 0.0600937736, 0.0680951666, 0.0771619327,
    0.0874359247, 0.0990778828, 0.1122699496, 0.1272185198,
    0.1441574689, 0.1633518128, 0.1851018540, 0.2097478796,
    0.2376754853, 0.2693216084, 0.3051813640, 0.3458157905,
    0.3918606281, 0.4440362645, 0.5031590062, 0.5701538495,
    0.6460689526, 0.7320920342, 0.8295689561, 0.9400247793
};

inline ed_decay_t ed_exp_pwl_lut_decay_fp(ed_accum_t exp_arg) {
    #pragma HLS INLINE off
    if (exp_arg >= ED_EXP_PWL_LUT_X_MAX) return (ed_decay_t)1;
    if (exp_arg < ED_EXP_PWL_LUT_X_MIN) return (ed_decay_t)0;

    ed_accum_t scaled_index =
        (exp_arg - ED_EXP_PWL_LUT_X_MIN) * ED_EXP_PWL_LUT_INV_STEP;
    int lut_idx = (int)scaled_index;
    if (lut_idx >= ED_EXP_PWL_LUT_SEGMENTS)
        lut_idx = ED_EXP_PWL_LUT_SEGMENTS - 1;

    ed_accum_t segment_x = ED_EXP_PWL_LUT_X_MIN
        + (ed_accum_t)lut_idx * ED_EXP_PWL_LUT_STEP;
    ed_accum_t local_x = exp_arg - segment_x;
    ed_accum_t result = (ed_accum_t)ED_EXP_PWL_LUT_VALUE[lut_idx]
        + ED_EXP_PWL_LUT_SLOPE[lut_idx] * local_x;

    if (result <= (ed_accum_t)0) return (ed_decay_t)0;
    if (result >= (ed_accum_t)1) return (ed_decay_t)1;
    return (ed_decay_t)result;
}

template<bool USE_PIECEWISE_LINEAR>
inline ed_decay_t ed_selected_decay_fp(ed_accum_t exp_arg) {
    (void)USE_PIECEWISE_LINEAR;
    return ed_exp_pwl_lut_decay_fp(exp_arg);
}

// =============================================================
// CUBA-LIF KERNEL — baseado em event_driven_srnn/cuba_lif.h
// =============================================================

typedef ed_decay_t (*ed_exp_decay_func_ptr)(ed_accum_t exp_arg);

static const ed_accum_t ED_LN_2 = (ed_accum_t)0.69314718055994530942;

static const ed_accum_t ED_LOG_PWL_VALUE[ED_LOG_PWL_SEGMENTS] = {
    0.000000000000, 0.030771658667, 0.060624621816, 0.089612158690,
    0.117783035656, 0.145182009844, 0.171850256927, 0.197825743330,
    0.223143551314, 0.247836163905, 0.271933715484, 0.295464212894,
    0.318453731119, 0.340926586971, 0.362905493689, 0.384411698910,
    0.405465108108, 0.426084395311, 0.446287102628, 0.466089729925,
    0.485507815782, 0.504556010752, 0.523248143765, 0.541597282433,
    0.559615787935, 0.577315365035, 0.594707107747, 0.611801541106,
    0.628608659422, 0.645137961374, 0.661398482245, 0.677398823592
};

static const ed_accum_t ED_LOG_PWL_SLOPE[ED_LOG_PWL_SEGMENTS] = {
    0.984693077336, 0.955294820790, 0.927601179944, 0.901468062934,
    0.876767174020, 0.853383906629, 0.831215564904, 0.810169855497,
    0.790163602892, 0.771121650530, 0.752975917126, 0.735664583190,
    0.719131387266, 0.703325015001, 0.688198567071, 0.673709094331,
    0.659817190488, 0.646486634161, 0.633684073478, 0.621378747427,
    0.609542239062, 0.598148256389, 0.587172437382, 0.576592176086,
    0.566386467181, 0.556535766780, 0.547021867498, 0.537827786124,
    0.528937662439, 0.520336667897, 0.512010923086, 0.503947422980
};

inline ed_accum_t ed_abs_accum(ed_accum_t value) {
    ed_accum_t result = value;
    if (value < (ed_accum_t)0)
        result = (ed_accum_t)0 - value;
    return result;
}

inline bool ed_tau_near_equal(ed_accum_t tau_mem, ed_accum_t tau_syn) {
    return ed_abs_accum(tau_mem - tau_syn) <= ED_INTERNAL_TIME_QUANTUM;
}

// Natural logarithm with base-2 range reduction and a 32-segment linear LUT.
// The loops have fixed bounds, which keeps the implementation synthesizable
// without relying on libm or a variable-latency normalization loop.
inline bool ed_log_pwl_fp(ed_accum_t x, ed_accum_t* result) {
    #pragma HLS INLINE off
    if (x <= (ed_accum_t)0) return false;

    ed_accum_t mantissa = x;
    int exponent = 0;
    ed_log_normalize:
    for (int iteration = 0; iteration < 32; ++iteration) {
        if (mantissa >= (ed_accum_t)2) {
            mantissa = mantissa * (ed_accum_t)0.5;
            ++exponent;
        } else if (mantissa < (ed_accum_t)1) {
            mantissa = mantissa * (ed_accum_t)2;
            --exponent;
        }
    }

    ed_accum_t scaled =
        (mantissa - (ed_accum_t)1) * (ed_accum_t)ED_LOG_PWL_SEGMENTS;
    int index = (int)scaled;
    if (index < 0) index = 0;
    if (index >= ED_LOG_PWL_SEGMENTS) index = ED_LOG_PWL_SEGMENTS - 1;

    ed_accum_t segment_start = (ed_accum_t)1
        + (ed_accum_t)index / (ed_accum_t)ED_LOG_PWL_SEGMENTS;
    ed_accum_t log_mantissa = ED_LOG_PWL_VALUE[index]
        + ED_LOG_PWL_SLOPE[index] * (mantissa - segment_start);
    *result = (ed_accum_t)exponent * ED_LN_2 + log_mantissa;
    return true;
}

template<bool USE_PIECEWISE_LINEAR>
inline ed_accum_t ed_cuba_lif_voltage_at(
    ed_accum_t v0,
    ed_accum_t i0,
    ed_accum_t dt,
    ed_accum_t r,
    ed_accum_t tau_mem,
    ed_accum_t tau_syn,
    ed_accum_t v_leak)
{
    #pragma HLS INLINE off
    if (dt <= (ed_accum_t)0) return v0;

    const ed_accum_t decay_mem =
        (ed_accum_t)ed_selected_decay_fp<USE_PIECEWISE_LINEAR>(
            -(dt / tau_mem));
    const ed_accum_t decay_syn =
        (ed_accum_t)ed_selected_decay_fp<USE_PIECEWISE_LINEAR>(
            -(dt / tau_syn));
    ed_accum_t voltage = v_leak + (v0 - v_leak) * decay_mem;
    const ed_accum_t tau_diff = tau_mem - tau_syn;
    if (!ed_tau_near_equal(tau_mem, tau_syn)) {
        voltage += (r * i0 * tau_syn / tau_diff)
            * (decay_mem - decay_syn);
    } else {
        const ed_accum_t tau = (tau_mem + tau_syn) * (ed_accum_t)0.5;
        const ed_accum_t decay =
            (ed_accum_t)ed_selected_decay_fp<USE_PIECEWISE_LINEAR>(
                -(dt / tau));
        voltage += (r * i0 / tau) * dt * decay;
    }
    return voltage;
}

template<bool USE_PIECEWISE_LINEAR>
inline void ed_advance_cuba_lif_state(
    ed_voltage_t* v_mem,
    ed_current_t* i_syn,
    ed_time_step_t dt,
    ed_accum_t r,
    ed_accum_t tau_mem,
    ed_accum_t tau_syn,
    ed_accum_t v_leak)
{
    #pragma HLS INLINE off
    if (dt <= (ed_time_step_t)0) return;
    const ed_accum_t dt_value = (ed_accum_t)dt;
    const ed_accum_t old_current = (ed_accum_t)(*i_syn);
    const ed_accum_t voltage =
        ed_cuba_lif_voltage_at<USE_PIECEWISE_LINEAR>(
        (ed_accum_t)(*v_mem), old_current, dt_value, r,
        tau_mem, tau_syn, v_leak);
    const ed_accum_t decay_syn =
        (ed_accum_t)ed_selected_decay_fp<USE_PIECEWISE_LINEAR>(
            -(dt_value / tau_syn));
    *v_mem = (ed_voltage_t)voltage;
    *i_syn = (ed_current_t)(old_current * decay_syn);
}

// Predict the first ascending threshold crossing after the current state.
// The prediction is valid only while no newer input reaches this neuron.
template<bool USE_PIECEWISE_LINEAR>
inline bool ed_predict_cuba_lif_spike(
    ed_voltage_t v_mem,
    ed_current_t i_syn,
    ed_accum_t r,
    ed_accum_t tau_mem,
    ed_accum_t tau_syn,
    ed_accum_t v_threshold,
    ed_accum_t v_leak,
    ed_time_step_t* crossing_dt)
{
    #pragma HLS INLINE off
    const ed_accum_t v0 = (ed_accum_t)v_mem;
    const ed_accum_t i0 = (ed_accum_t)i_syn;
    if (v0 >= v_threshold) {
        *crossing_dt = (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM;
        return true;
    }

    const ed_accum_t a = v0 - v_leak;
    const ed_accum_t drive = r * i0;
    // dV/dt at zero is proportional to drive - a.  With a non-positive
    // drive, or a state already descending, no future excitatory peak exists.
    if (drive <= (ed_accum_t)0 || drive <= a) return false;

    const ed_accum_t tau_diff = tau_mem - tau_syn;
    ed_accum_t peak_dt = 0;
    if (ed_tau_near_equal(tau_mem, tau_syn)) {
        const ed_accum_t tau = (tau_mem + tau_syn) * (ed_accum_t)0.5;
        peak_dt = tau * ((ed_accum_t)1 - a / drive);
    } else {
        const ed_accum_t denominator = a * tau_diff + drive * tau_syn;
        if (denominator <= (ed_accum_t)0) return false;
        const ed_accum_t ratio = drive * tau_mem / denominator;
        ed_accum_t log_ratio = 0;
        if (!ed_log_pwl_fp(ratio, &log_ratio)) return false;
        peak_dt = (tau_mem * tau_syn / tau_diff) * log_ratio;
    }
    if (peak_dt <= (ed_accum_t)0) return false;

    const ed_accum_t peak_voltage =
        ed_cuba_lif_voltage_at<USE_PIECEWISE_LINEAR>(
            v0, i0, peak_dt, r, tau_mem, tau_syn, v_leak);
    if (peak_voltage < v_threshold) return false;

    ed_accum_t low = 0;
    ed_accum_t high = peak_dt;
    ed_crossing_bisection:
    for (int iteration = 0; iteration < ED_ROOT_BISECTION_ITERATIONS;
         ++iteration) {
        const ed_accum_t middle = (low + high) * (ed_accum_t)0.5;
        const ed_accum_t middle_voltage =
            ed_cuba_lif_voltage_at<USE_PIECEWISE_LINEAR>(
                v0, i0, middle, r, tau_mem, tau_syn, v_leak);
        if (middle_voltage >= v_threshold)
            high = middle;
        else
            low = middle;
    }

    // Conversion to the internal time type must be causal: never schedule
    // earlier than the first root.  Real ap_fixed truncates here; the second
    // branch is a no-op in the lightweight host test stub.
    ed_time_step_t quantized = (ed_time_step_t)high;
    if ((ed_accum_t)quantized < high)
        quantized += (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM;
    if (quantized < (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM)
        quantized = (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM;
    *crossing_dt = quantized;
    return true;
}

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
    ed_exp_decay_func_ptr exp_decay_func,
    bool reset_by_subtraction = false
) {
    #pragma HLS INLINE off
    const ed_accum_t dt_f   = (ed_accum_t)dt;
    const ed_accum_t tau_m_raw = (ed_accum_t)tau_mem;
    const ed_accum_t tau_s_raw = (ed_accum_t)tau_syn;
    // A low-precision generated weight type can round a valid time constant
    // to zero.  Clamp it before division so CSim/HLS cannot raise SIGFPE.
    // The generator normally emits at least ap_fixed<24,8> for Cuba-LIF;
    // this guard also protects projects generated with older settings.
    const ed_accum_t min_tau = (ed_accum_t)(1.0 / 65536.0);
    const ed_accum_t tau_m = tau_m_raw > (ed_accum_t)0 ? tau_m_raw : min_tau;
    const ed_accum_t tau_s = tau_s_raw > (ed_accum_t)0 ? tau_s_raw : min_tau;
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
        *v_mem = reset_by_subtraction
            ? (ed_voltage_t)(v_t - (ed_accum_t)v_th)
            : v_reset;
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
        ed_spike_t s = in.read();
        out1.write(s);
        out2.write(s);
        if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE)
            step_done = true;
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
    ed_spike_t first = in1.read();
    ed_spike_t second = in2.read();
    bool first_done = first.type != ED_TYPE_SPIKE;
    bool second_done = second.type != ED_TYPE_SPIKE;

    // Both producers emit nondecreasing timestamps followed by a watermark.
    // Merge them as two sorted finite streams so fractional self-event times
    // survive fan-in and recurrent paths.
    while (!first_done || !second_done) {
        bool take_first = !first_done
            && (second_done || first.timestamp <= second.timestamp);
        if (take_first) {
            out.write(first);
            first = in1.read();
            first_done = first.type != ED_TYPE_SPIKE;
        } else {
            out.write(second);
            second = in2.read();
            second_done = second.type != ED_TYPE_SPIKE;
        }
    }

    ed_spike_t end_token = first.timestamp >= second.timestamp ? first : second;
    if (first.type == ED_TYPE_END_SAMPLE || second.type == ED_TYPE_END_SAMPLE)
        end_token.type = ED_TYPE_END_SAMPLE;
    else
        end_token.type = ED_TYPE_END_STEP;
    out.write(end_token);
}

// =============================================================
// LINEAR — baseado em event_driven_srnn/linear.h
// =============================================================

template<int NUM_OUTPUTS>
inline void ed_emit_accumulated_vector(
    hls::stream<ed_spike_t>& output_stream,
    const ed_spike_t& group_event,
    ed_accum_t accum[NUM_OUTPUTS])
{
    for (int out_idx = 0; out_idx < NUM_OUTPUTS; ++out_idx) {
        #pragma HLS PIPELINE II=1
        if (accum[out_idx] != (ed_accum_t)0) {
            ed_spike_t out_spike = group_event;
            out_spike.type = ED_TYPE_SPIKE;
            out_spike.amplitude = (ed_current_t)accum[out_idx];
            out_spike.channel_idx = 0;
            out_spike.height_idx = 0;
            out_spike.width_idx = out_idx;
            output_stream.write(out_spike);
        }
        accum[out_idx] = 0;
    }
}

template<int NUM_INPUTS, int NUM_OUTPUTS,
         typename params_type>
void Linear(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const params_type weights[NUM_OUTPUTS][NUM_INPUTS])
{
        ed_accum_t acc_buffer[NUM_OUTPUTS] = {};
    ed_spike_t group_event = {};
    bool group_valid = false;

    bool end_of_processing = false;
    main_loop: while (!end_of_processing) {
        ed_spike_t s = input_stream.read();

        if (s.type == ED_TYPE_SPIKE) {
            if (group_valid && s.timestamp != group_event.timestamp) {
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, acc_buffer);
                group_valid = false;
            }
            if (!group_valid) {
                group_event = s;
                group_valid = true;
            }
            unsigned int in_idx = s.width_idx;
            ed_current_t amp = s.amplitude;
            if (in_idx < NUM_INPUTS) {
                update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; ++out_idx) {
                    #pragma HLS PIPELINE II=1
                    acc_buffer[out_idx] +=
                        (ed_accum_t)weights[out_idx][in_idx] * (ed_accum_t)amp;
                }
            }
        } else if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE) {
            if (group_valid)
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, acc_buffer);
            output_stream.write(s);
            end_of_processing = true;
        }
    }
}

// =============================================================
// AFFINE — baseado em event_driven_srnn/affine.h
// =============================================================

template<int NUM_INPUTS, int NUM_OUTPUTS,
         typename params_type>
void Affine(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    params_type weights[NUM_OUTPUTS][NUM_INPUTS],
    params_type biases[NUM_OUTPUTS],
    bool reset_timing = false)
{
        static ed_time_step_t interval_start = 0;
    if (reset_timing) interval_start = 0;

    ed_accum_t acc_buffer[NUM_OUTPUTS];
    affine_bias_init:
    for (int out_idx = 0; out_idx < NUM_OUTPUTS; ++out_idx) {
        #pragma HLS PIPELINE II=1
        acc_buffer[out_idx] = (ed_accum_t)biases[out_idx];
    }
    ed_spike_t group_event = {};
    group_event.type = ED_TYPE_SPIKE;
    group_event.timestamp = interval_start;
    bool group_valid = true;
    bool metadata_initialized = false;

    bool end_of_processing = false;
    main_loop: while (!end_of_processing) {
        ed_spike_t s = input_stream.read();

        if (!metadata_initialized) {
            group_event.time_step = s.time_step;
            group_event.batch_idx = s.batch_idx;
            metadata_initialized = true;
        }

        if (s.type == ED_TYPE_SPIKE) {
            if (group_valid && s.timestamp != group_event.timestamp) {
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, acc_buffer);
                group_valid = false;
            }
            if (!group_valid) {
                group_event = s;
                group_valid = true;
            }
            unsigned int in_idx  = s.width_idx;
            ed_current_t amp     = s.amplitude;
            if (in_idx < NUM_INPUTS) {
                update_loop: for (int out_idx = 0; out_idx < NUM_OUTPUTS; ++out_idx) {
                    #pragma HLS PIPELINE II=1
                    acc_buffer[out_idx] +=
                        (ed_accum_t)weights[out_idx][in_idx] * (ed_accum_t)amp;
                }
            }
        } else if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE) {
            if (group_valid)
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, acc_buffer);
            output_stream.write(s);
            interval_start = s.type == ED_TYPE_END_SAMPLE
                ? (ed_time_step_t)0 : s.timestamp;
            end_of_processing = true;
        }
    }
}

// One-watermark accumulation used on explicitly delayed recurrent edges.
// The reference time-driven graph stores one recurrent vector per logical
// step; keeping that edge step-based avoids an unbounded temporal expansion
// of a dense recurrent matrix while feed-forward edges retain self-event
// timestamps.
template<int NUM_INPUTS, int NUM_OUTPUTS,
         typename params_type>
void LinearStep(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const params_type weights[NUM_OUTPUTS][NUM_INPUTS])
{
        ed_accum_t accum[NUM_OUTPUTS] = {};
    while (true) {
        ed_spike_t event = input_stream.read();
        if (event.type == ED_TYPE_SPIKE) {
            const unsigned int input_index = event.width_idx;
            if (input_index < NUM_INPUTS) {
                for (int output_index = 0;
                     output_index < NUM_OUTPUTS; ++output_index) {
                    #pragma HLS PIPELINE II=1
                    accum[output_index] +=
                        (ed_accum_t)weights[output_index][input_index]
                        * (ed_accum_t)event.amplitude;
                }
            }
        } else {
            ed_emit_accumulated_vector<NUM_OUTPUTS>(
                output_stream, event, accum);
            output_stream.write(event);
            break;
        }
    }
}

template<int NUM_INPUTS, int NUM_OUTPUTS,
         typename params_type>
void AffineStep(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    params_type weights[NUM_OUTPUTS][NUM_INPUTS],
    params_type biases[NUM_OUTPUTS],
    bool reset_timing = false)
{
        (void)reset_timing;
    ed_accum_t accum[NUM_OUTPUTS];
    for (int output_index = 0; output_index < NUM_OUTPUTS; ++output_index)
        accum[output_index] = (ed_accum_t)biases[output_index];

    while (true) {
        ed_spike_t event = input_stream.read();
        if (event.type == ED_TYPE_SPIKE) {
            const unsigned int input_index = event.width_idx;
            if (input_index < NUM_INPUTS) {
                for (int output_index = 0;
                     output_index < NUM_OUTPUTS; ++output_index) {
                    #pragma HLS PIPELINE II=1
                    accum[output_index] +=
                        (ed_accum_t)weights[output_index][input_index]
                        * (ed_accum_t)event.amplitude;
                }
            }
        } else {
            ed_emit_accumulated_vector<NUM_OUTPUTS>(
                output_stream, event, accum);
            output_stream.write(event);
            break;
        }
    }
}

// =============================================================
// SPARSE SYNAPTIC TABLES (CSC, input-major)
// =============================================================

// One incoming spike selects exactly one CSC column.  Only its stored
// non-zero synapses are read and accumulated; no dense output scan occurs on
// the synaptic update path.
template<int NUM_INPUTS, int NUM_OUTPUTS, int NUM_NONZERO,
         typename index_type, typename params_type>
inline void ed_sparse_accumulate_column(
    const ed_spike_t& event,
    ed_accum_t accum[NUM_OUTPUTS],
    const index_type column_offsets[NUM_INPUTS + 1],
    const index_type row_indices[NUM_NONZERO],
    const params_type values[NUM_NONZERO])
{
    #pragma HLS INLINE off
    const unsigned int input_index = event.width_idx;
    if (input_index >= (unsigned int)NUM_INPUTS) return;
    const unsigned int first = (unsigned int)column_offsets[input_index];
    const unsigned int last = (unsigned int)column_offsets[input_index + 1];
    ed_sparse_column_loop:
    for (unsigned int position = first; position < last; ++position) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=NUM_NONZERO
        #pragma HLS PIPELINE II=1
        const unsigned int output_index =
            (unsigned int)row_indices[position];
        if (output_index < (unsigned int)NUM_OUTPUTS) {
            accum[output_index] +=
                (ed_accum_t)values[position]
                * (ed_accum_t)event.amplitude;
        }
    }
}

template<int NUM_INPUTS, int NUM_OUTPUTS, int NUM_NONZERO,
         typename index_type, typename params_type>
void LinearSparse(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const index_type column_offsets[NUM_INPUTS + 1],
    const index_type row_indices[NUM_NONZERO],
    const params_type values[NUM_NONZERO])
{
        ed_accum_t accum[NUM_OUTPUTS] = {};
    ed_spike_t group_event = {};
    bool group_valid = false;

    while (true) {
        ed_spike_t event = input_stream.read();
        if (event.type == ED_TYPE_SPIKE) {
            if (group_valid && event.timestamp != group_event.timestamp) {
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, accum);
                group_valid = false;
            }
            if (!group_valid) {
                group_event = event;
                group_valid = true;
            }
            ed_sparse_accumulate_column<
                NUM_INPUTS, NUM_OUTPUTS, NUM_NONZERO
            >(event, accum, column_offsets, row_indices, values);
        } else {
            if (group_valid)
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, accum);
            output_stream.write(event);
            break;
        }
    }
}

template<int NUM_INPUTS, int NUM_OUTPUTS, int NUM_NONZERO,
         typename index_type, typename params_type>
void AffineSparse(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const index_type column_offsets[NUM_INPUTS + 1],
    const index_type row_indices[NUM_NONZERO],
    const params_type values[NUM_NONZERO],
    const params_type biases[NUM_OUTPUTS],
    bool reset_timing = false)
{
        static ed_time_step_t interval_start = 0;
    if (reset_timing) interval_start = 0;

    ed_accum_t accum[NUM_OUTPUTS];
    for (int output_index = 0; output_index < NUM_OUTPUTS; ++output_index)
        accum[output_index] = (ed_accum_t)biases[output_index];
    ed_spike_t group_event = {};
    group_event.type = ED_TYPE_SPIKE;
    group_event.timestamp = interval_start;
    bool group_valid = true;
    bool metadata_initialized = false;

    while (true) {
        ed_spike_t event = input_stream.read();
        if (!metadata_initialized) {
            group_event.time_step = event.time_step;
            group_event.batch_idx = event.batch_idx;
            metadata_initialized = true;
        }
        if (event.type == ED_TYPE_SPIKE) {
            if (group_valid && event.timestamp != group_event.timestamp) {
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, accum);
                group_valid = false;
            }
            if (!group_valid) {
                group_event = event;
                group_valid = true;
            }
            ed_sparse_accumulate_column<
                NUM_INPUTS, NUM_OUTPUTS, NUM_NONZERO
            >(event, accum, column_offsets, row_indices, values);
        } else {
            if (group_valid)
                ed_emit_accumulated_vector<NUM_OUTPUTS>(
                    output_stream, group_event, accum);
            output_stream.write(event);
            interval_start = event.type == ED_TYPE_END_SAMPLE
                ? (ed_time_step_t)0 : event.timestamp;
            break;
        }
    }
}

template<int NUM_INPUTS, int NUM_OUTPUTS, int NUM_NONZERO,
         typename index_type, typename params_type>
void LinearSparseStep(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const index_type column_offsets[NUM_INPUTS + 1],
    const index_type row_indices[NUM_NONZERO],
    const params_type values[NUM_NONZERO])
{
        ed_accum_t accum[NUM_OUTPUTS] = {};
    while (true) {
        ed_spike_t event = input_stream.read();
        if (event.type == ED_TYPE_SPIKE) {
            ed_sparse_accumulate_column<
                NUM_INPUTS, NUM_OUTPUTS, NUM_NONZERO
            >(event, accum, column_offsets, row_indices, values);
        } else {
            ed_emit_accumulated_vector<NUM_OUTPUTS>(
                output_stream, event, accum);
            output_stream.write(event);
            break;
        }
    }
}

template<int NUM_INPUTS, int NUM_OUTPUTS, int NUM_NONZERO,
         typename index_type, typename params_type>
void AffineSparseStep(
    hls::stream<ed_spike_t>& input_stream,
    hls::stream<ed_spike_t>& output_stream,
    const index_type column_offsets[NUM_INPUTS + 1],
    const index_type row_indices[NUM_NONZERO],
    const params_type values[NUM_NONZERO],
    const params_type biases[NUM_OUTPUTS],
    bool reset_timing = false)
{
        (void)reset_timing;
    ed_accum_t accum[NUM_OUTPUTS];
    for (int output_index = 0; output_index < NUM_OUTPUTS; ++output_index)
        accum[output_index] = (ed_accum_t)biases[output_index];

    while (true) {
        ed_spike_t event = input_stream.read();
        if (event.type == ED_TYPE_SPIKE) {
            ed_sparse_accumulate_column<
                NUM_INPUTS, NUM_OUTPUTS, NUM_NONZERO
            >(event, accum, column_offsets, row_indices, values);
        } else {
            ed_emit_accumulated_vector<NUM_OUTPUTS>(
                output_stream, event, accum);
            output_stream.write(event);
            break;
        }
    }
}

// =============================================================
// CUBA-LIF WRAPPER — baseado em event_driven_srnn/cuba_lif.h
// =============================================================

template<int IN_C, int IN_H, int IN_W, int INSTANCE_ID = 0,
         bool USE_PIECEWISE_LINEAR = true,
         bool RESET_BY_SUBTRACTION = false,
         bool ALLOW_MULTIPLE_SELF_EVENTS = false,
         typename tau_type, typename params_type>
// Legacy scheduled-self-event primitive retained only for source compatibility;
// the NeuroHLS generator no longer emits or selects this path. New designs use
// CubaLIFActiveList exclusively.
void CubaLIF(
    hls::stream<ed_spike_t>& in,
    hls::stream<ed_spike_t>& out,
    tau_type tau_syn[IN_W],
    tau_type tau_mem[IN_W],
    params_type r[IN_W],
    params_type v_leak[IN_W],
    params_type v_threshold[IN_W],
    params_type v_reset[IN_W],
    params_type w_in[IN_W],
    bool reset_potentials = false)
{
        static ed_voltage_t  v_mem[IN_C][IN_H][IN_W] = {};
    static ed_current_t  i_syn[IN_C][IN_H][IN_W] = {};
    static ed_time_step_t state_time[IN_C][IN_H][IN_W] = {};
    // Indexed event calendar: each neuron owns at most one prediction.  A
    // min scan is smaller than a cancellable heap for the current 7--40
    // neuron layers and provides the same priority-queue semantics.
    static ed_time_step_t scheduled_time[IN_C][IN_H][IN_W] = {};
    static bool scheduled_valid[IN_C][IN_H][IN_W] = {};
    // The time-driven reference can emit at most one spike per neuron and
    // logical step.  Keep that refractory contract while locating the spike
    // itself at sub-step precision.
    bool fired_in_window[IN_C][IN_H][IN_W] = {};

    if (reset_potentials) {
        for (int c = 0; c < IN_C; c++)
            for (int h = 0; h < IN_H; h++)
                for (int w = 0; w < IN_W; w++) {
                    v_mem[c][h][w] = v_reset[w];
                    i_syn[c][h][w] = 0;
                    state_time[c][h][w] = 0;
                    scheduled_time[c][h][w] = 0;
                    scheduled_valid[c][h][w] = false;
                }
    }

    bool end_of_processing = false;
    while (!end_of_processing) {
        ed_spike_t s = in.read();

        // The current input timestamp (or END_* watermark) guarantees that
        // no earlier external event remains in this stream.  Release every
        // predicted self-event due up to that time in global timestamp order.
        const int max_self_events = IN_C * IN_H * IN_W
            * (ALLOW_MULTIPLE_SELF_EVENTS
                ? ED_MAX_SELF_EVENTS_PER_NEURON_PER_WINDOW : 1);
        ed_release_self_events:
        for (int event_count = 0; event_count < max_self_events; ++event_count) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=max_self_events
            bool found = false;
            ed_time_step_t earliest = s.timestamp;
            int fire_c = 0;
            int fire_h = 0;
            int fire_w = 0;

            const int total_neurons = IN_C * IN_H * IN_W;
            ed_calendar_scan_c:
            for (int index = 0; index < total_neurons; ++index) {
                const int fire_w_candidate = index % IN_W;
                const int spatial = index / IN_W;
                const int fire_h_candidate = spatial % IN_H;
                const int fire_c_candidate = spatial / IN_H;
                if (scheduled_valid[fire_c_candidate][fire_h_candidate][fire_w_candidate]
                    && scheduled_time[fire_c_candidate][fire_h_candidate][fire_w_candidate]
                        <= s.timestamp
                    && (!found || scheduled_time[
                            fire_c_candidate][fire_h_candidate][fire_w_candidate]
                        < earliest)) {
                    // A strict comparison preserves the original flattened
                    // c,h,w tie-breaking order.
                    earliest = scheduled_time[
                        fire_c_candidate][fire_h_candidate][fire_w_candidate];
                    fire_w = fire_w_candidate;
                    fire_h = fire_h_candidate;
                    fire_c = fire_c_candidate;
                    found = true;
                }
            }
            if (!found) break;

            scheduled_valid[fire_c][fire_h][fire_w] = false;
            const ed_accum_t tau_m_raw = (ed_accum_t)tau_mem[fire_w];
            const ed_accum_t tau_s_raw = (ed_accum_t)tau_syn[fire_w];
            const ed_accum_t tau_m = tau_m_raw > ED_INTERNAL_TIME_QUANTUM
                ? tau_m_raw : ED_INTERNAL_TIME_QUANTUM;
            const ed_accum_t tau_s = tau_s_raw > ED_INTERNAL_TIME_QUANTUM
                ? tau_s_raw : ED_INTERNAL_TIME_QUANTUM;
            ed_time_step_t elapsed = 0;
            if (earliest > state_time[fire_c][fire_h][fire_w])
                elapsed = earliest - state_time[fire_c][fire_h][fire_w];
            ed_advance_cuba_lif_state<USE_PIECEWISE_LINEAR>(
                &v_mem[fire_c][fire_h][fire_w],
                &i_syn[fire_c][fire_h][fire_w], elapsed,
                (ed_accum_t)r[fire_w], tau_m, tau_s,
                (ed_accum_t)v_leak[fire_w]);
            state_time[fire_c][fire_h][fire_w] = earliest;

            const ed_accum_t pre_reset = (ed_accum_t)v_mem[fire_c][fire_h][fire_w];
            if (RESET_BY_SUBTRACTION) {
                const ed_accum_t threshold = (ed_accum_t)v_threshold[fire_w];
                v_mem[fire_c][fire_h][fire_w] = (ed_voltage_t)(
                    (pre_reset > threshold ? pre_reset : threshold) - threshold);
            } else {
                v_mem[fire_c][fire_h][fire_w] = v_reset[fire_w];
            }

            ed_spike_t out_spike = s;
            out_spike.type = ED_TYPE_SPIKE;
            out_spike.amplitude = (ed_current_t)1;
            out_spike.timestamp = earliest;
            out_spike.channel_idx = fire_c;
            out_spike.height_idx = fire_h;
            out_spike.width_idx = fire_w;
            out.write(out_spike);
            fired_in_window[fire_c][fire_h][fire_w] = true;

            // In the physical continuous mode the synaptic current survives
            // reset and may create another crossing before the next external
            // event.  Reinsert that crossing immediately.  The
            // discrete-compatible mode deliberately keeps the source
            // network's one-spike-per-step refractory contract.
            if (ALLOW_MULTIPLE_SELF_EVENTS) {
                ed_time_step_t next_delta = 0;
                if (ed_predict_cuba_lif_spike<USE_PIECEWISE_LINEAR>(
                        v_mem[fire_c][fire_h][fire_w],
                        i_syn[fire_c][fire_h][fire_w],
                        (ed_accum_t)r[fire_w], tau_m, tau_s,
                        (ed_accum_t)v_threshold[fire_w],
                        (ed_accum_t)v_leak[fire_w], &next_delta)) {
                    if (next_delta
                        < (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM)
                        next_delta =
                            (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM;
                    scheduled_time[fire_c][fire_h][fire_w] =
                        earliest + next_delta;
                    scheduled_valid[fire_c][fire_h][fire_w] = true;
                }
            }
        }

        if (s.type == ED_TYPE_SPIKE) {
            uint16_t c = s.channel_idx;
            uint16_t h = s.height_idx;
            uint16_t w = s.width_idx;
            if (c < IN_C && h < IN_H && w < IN_W) {
                // A malformed or unsupported upstream path must never move a
                // neuron's state backwards in time.  Generated paths are
                // sorted; hardware clamps a stale timestamp to the current
                // state time as a final safety net.
                const ed_time_step_t event_time =
                    s.timestamp < state_time[c][h][w]
                    ? state_time[c][h][w] : s.timestamp;
                const ed_accum_t tau_m_raw = (ed_accum_t)tau_mem[w];
                const ed_accum_t tau_s_raw = (ed_accum_t)tau_syn[w];
                const ed_accum_t tau_m = tau_m_raw > ED_INTERNAL_TIME_QUANTUM
                    ? tau_m_raw : ED_INTERNAL_TIME_QUANTUM;
                const ed_accum_t tau_s = tau_s_raw > ED_INTERNAL_TIME_QUANTUM
                    ? tau_s_raw : ED_INTERNAL_TIME_QUANTUM;
                ed_time_step_t elapsed = 0;
                if (event_time > state_time[c][h][w])
                    elapsed = event_time - state_time[c][h][w];
                ed_advance_cuba_lif_state<USE_PIECEWISE_LINEAR>(
                    &v_mem[c][h][w], &i_syn[c][h][w], elapsed,
                    (ed_accum_t)r[w], tau_m, tau_s,
                    (ed_accum_t)v_leak[w]);
                state_time[c][h][w] = event_time;

                // Any newer input invalidates the previous prediction for
                // this neuron.  Voltage is continuous; only current jumps.
                scheduled_valid[c][h][w] = false;
                i_syn[c][h][w] = (ed_current_t)(
                    (ed_accum_t)i_syn[c][h][w]
                    + (ed_accum_t)w_in[w] * (ed_accum_t)s.amplitude);

                ed_time_step_t crossing_delta = 0;
                if ((ALLOW_MULTIPLE_SELF_EVENTS
                        || !fired_in_window[c][h][w])
                    && ed_predict_cuba_lif_spike<USE_PIECEWISE_LINEAR>(
                        v_mem[c][h][w], i_syn[c][h][w],
                        (ed_accum_t)r[w], tau_m, tau_s,
                        (ed_accum_t)v_threshold[w],
                        (ed_accum_t)v_leak[w],
                        &crossing_delta)) {
                    if (crossing_delta < (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM)
                        crossing_delta = (ed_time_step_t)ED_INTERNAL_TIME_QUANTUM;
                    scheduled_time[c][h][w] = event_time + crossing_delta;
                    scheduled_valid[c][h][w] = true;
                }
            }

        } else if (s.type == ED_TYPE_END_STEP || s.type == ED_TYPE_END_SAMPLE) {
            // A neuron that fired remains refractory until this watermark.
            // Advance its residual current through the rest of the interval,
            // then predict the first crossing of the next interval.
            if (!ALLOW_MULTIPLE_SELF_EVENTS
                && s.type == ED_TYPE_END_STEP) {
                for (int c = 0; c < IN_C; ++c) {
                    for (int h = 0; h < IN_H; ++h) {
                        for (int w = 0; w < IN_W; ++w) {
                            if (!fired_in_window[c][h][w]) continue;
                            const ed_accum_t tau_m_raw = (ed_accum_t)tau_mem[w];
                            const ed_accum_t tau_s_raw = (ed_accum_t)tau_syn[w];
                            const ed_accum_t tau_m = tau_m_raw > ED_INTERNAL_TIME_QUANTUM
                                ? tau_m_raw : ED_INTERNAL_TIME_QUANTUM;
                            const ed_accum_t tau_s = tau_s_raw > ED_INTERNAL_TIME_QUANTUM
                                ? tau_s_raw : ED_INTERNAL_TIME_QUANTUM;
                            ed_time_step_t elapsed = 0;
                            if (s.timestamp > state_time[c][h][w])
                                elapsed = s.timestamp - state_time[c][h][w];
                            ed_advance_cuba_lif_state<USE_PIECEWISE_LINEAR>(
                                &v_mem[c][h][w], &i_syn[c][h][w], elapsed,
                                (ed_accum_t)r[w], tau_m, tau_s,
                                (ed_accum_t)v_leak[w]);
                            state_time[c][h][w] = s.timestamp;

                            ed_time_step_t next_delta = 0;
                            if (ed_predict_cuba_lif_spike<USE_PIECEWISE_LINEAR>(
                                    v_mem[c][h][w], i_syn[c][h][w],
                                    (ed_accum_t)r[w], tau_m, tau_s,
                                    (ed_accum_t)v_threshold[w],
                                    (ed_accum_t)v_leak[w],
                                    &next_delta)) {
                                scheduled_time[c][h][w] = s.timestamp + next_delta;
                                scheduled_valid[c][h][w] = true;
                            }
                        }
                    }
                }
            }
            out.write(s);
            end_of_processing = true;

            if (s.type == ED_TYPE_END_SAMPLE) {
                for (int c = 0; c < IN_C; c++)
                    for (int h = 0; h < IN_H; h++)
                        for (int w = 0; w < IN_W; w++) {
                            v_mem[c][h][w] = v_reset[w];
                            i_syn[c][h][w] = 0;
                            state_time[c][h][w] = 0;
                            scheduled_time[c][h][w] = 0;
                            scheduled_valid[c][h][w] = false;
                        }
            }
            break;
        }
    }
}

// =============================================================
// CUBA-LIF ACTIVE LIST / LIGHTWEIGHT TICKS
// =============================================================

// Multiply by a coefficient represented as a sum of negative powers of two.
// Each shift is stored in ROM with the neuron parameters; the tick datapath is
// therefore limited to arithmetic shifts and additions.
template<typename shift_type, typename count_type>
inline ed_accum_t ed_active_shift_scale(
    ed_accum_t value,
    const shift_type shifts[ED_ACTIVE_SHIFT_TERMS],
    count_type term_count)
{
    #pragma HLS INLINE
    ed_accum_t result = 0;
    const unsigned int count = (unsigned int)term_count;
    ed_active_shift_terms:
    for (int term = 0; term < ED_ACTIVE_SHIFT_TERMS; ++term) {
        if ((unsigned int)term < count)
            result += value >> (unsigned int)shifts[term];
    }
    return result;
}

// Hybrid event/tick implementation.  Synaptic events only update the target
// neuron's pending drive and active-list membership.  END_STEP/END_SAMPLE is
// the lightweight tick and visits only active IDs.
//
// The persisted current is beta_mem * R * u.  Folding beta_mem * R into the
// event-only input gain makes the membrane tick
//
//   drive' = (1 - alpha_syn) * drive + pending_drive
//   v'     = v - beta_mem * (v - v_leak) + drive'
//
// equivalent to the Euler CUBA-LIF recurrence while keeping the tick itself
// multiplier-free.  Pending input is added after old-current decay, so a new
// event is not accidentally decayed in the same interval.
template<int IN_C, int IN_H, int IN_W, int INSTANCE_ID = 0,
         bool RESET_BY_SUBTRACTION = false,
         typename shift_type, typename count_type, typename params_type>
void CubaLIFActiveList(
    hls::stream<ed_spike_t>& in,
    hls::stream<ed_spike_t>& out,
    const shift_type decay_u_shifts[IN_W][ED_ACTIVE_SHIFT_TERMS],
    const count_type decay_u_terms[IN_W],
    const shift_type decay_v_shifts[IN_W][ED_ACTIVE_SHIFT_TERMS],
    const count_type decay_v_terms[IN_W],
    const params_type v_leak[IN_W],
    const params_type v_threshold[IN_W],
    const params_type v_reset[IN_W],
    const params_type input_gain[IN_W],
    ed_accum_t noise_threshold,
    bool reset_potentials = false)
{
        const int TOTAL_NEURONS = IN_C * IN_H * IN_W;
    static ed_voltage_t v_mem[TOTAL_NEURONS] = {};
    static ed_current_t current_drive[TOTAL_NEURONS] = {};
    static ed_accum_t pending_drive[TOTAL_NEURONS] = {};
    static unsigned int active_ids[TOTAL_NEURONS] = {};
    static unsigned int next_active_ids[TOTAL_NEURONS] = {};
    static bool active_flag[TOTAL_NEURONS] = {};
    static unsigned int active_count = 0;

    if (reset_potentials) {
        active_count = 0;
        ed_active_reset:
        for (int neuron = 0; neuron < TOTAL_NEURONS; ++neuron) {
            const int w = (IN_C == 1 && IN_H == 1)
                ? neuron : neuron % IN_W;
            v_mem[neuron] = v_reset[w];
            current_drive[neuron] = 0;
            pending_drive[neuron] = 0;
            active_ids[neuron] = 0;
            next_active_ids[neuron] = 0;
            const bool starts_active = ed_abs_accum(
                (ed_accum_t)v_reset[w] - (ed_accum_t)v_leak[w]
            ) > noise_threshold;
            active_flag[neuron] = starts_active;
            if (starts_active)
                active_ids[active_count++] = neuron;
        }
    }

    bool finished = false;
    while (!finished) {
        ed_spike_t event = in.read();
        if (event.type == ED_TYPE_SPIKE) {
            const unsigned int c = event.channel_idx;
            const unsigned int h = event.height_idx;
            const unsigned int w = event.width_idx;
            if (c < (unsigned int)IN_C
                && h < (unsigned int)IN_H
                && w < (unsigned int)IN_W) {
                const unsigned int neuron =
                    (c * (unsigned int)IN_H + h) * (unsigned int)IN_W + w;
                pending_drive[neuron] +=
                    (ed_accum_t)input_gain[w] * (ed_accum_t)event.amplitude;

                // Magnitude is used instead of u > 0 so inhibitory events are
                // retained as first-class state transitions.
                // Every representable contribution must be consumed at this
                // step's watermark.  Epsilon is a post-tick removal rule; if
                // it were also used here, a sub-epsilon pending value could
                // survive silent ticks and combine with a much later event.
                if (!active_flag[neuron]
                    && pending_drive[neuron] != (ed_accum_t)0
                    && active_count < (unsigned int)TOTAL_NEURONS) {
                    active_ids[active_count++] = neuron;
                    active_flag[neuron] = true;
                }
            }
            continue;
        }

        if (event.type == ED_TYPE_END_STEP
            || event.type == ED_TYPE_END_SAMPLE) {
            const unsigned int old_active_count = active_count;
            unsigned int next_active_count = 0;
            bool tick_spike[TOTAL_NEURONS] = {};
            bool tick_survivor[TOTAL_NEURONS] = {};

            // The runtime bound is the active-list snapshot, so an empty or
            // short list does not spend cycles scanning inactive neurons.  A
            // static tripcount still gives HLS the required worst-case bound.
            // The current list is read-only during the pipelined state update;
            // survivors are compacted into a distinct buffer.  Active IDs are
            // unique by construction (active_flag), so state accesses from
            // different loop iterations do not alias.
            #pragma HLS DEPENDENCE variable=current_drive inter false
            #pragma HLS DEPENDENCE variable=pending_drive inter false
            #pragma HLS DEPENDENCE variable=v_mem inter false
            #pragma HLS DEPENDENCE variable=active_flag inter false
            ed_active_tick:
            for (unsigned int slot = 0; slot < old_active_count; ++slot) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=TOTAL_NEURONS
                #pragma HLS PIPELINE II=1
                const unsigned int neuron = active_ids[slot];
                unsigned int w = neuron;
                unsigned int h = 0;
                unsigned int c = 0;
                if (IN_C != 1 || IN_H != 1) {
                    w = neuron % (unsigned int)IN_W;
                    const unsigned int spatial = neuron / (unsigned int)IN_W;
                    h = spatial % (unsigned int)IN_H;
                    c = spatial / (unsigned int)IN_H;
                }

                const ed_accum_t old_drive = (ed_accum_t)current_drive[neuron];
                const ed_accum_t drive_decay = ed_active_shift_scale(
                    old_drive, decay_u_shifts[w], decay_u_terms[w]);
                ed_accum_t next_drive = old_drive - drive_decay
                    + pending_drive[neuron];
                pending_drive[neuron] = 0;

                const ed_accum_t old_voltage = (ed_accum_t)v_mem[neuron];
                const ed_accum_t centered_voltage =
                    old_voltage - (ed_accum_t)v_leak[w];
                const ed_accum_t voltage_decay = ed_active_shift_scale(
                    centered_voltage, decay_v_shifts[w], decay_v_terms[w]);
                ed_accum_t next_voltage = old_voltage - voltage_decay + next_drive;

                tick_spike[slot] = next_voltage >= (ed_accum_t)v_threshold[w];
                if (tick_spike[slot]) {
                    if (RESET_BY_SUBTRACTION)
                        next_voltage -= (ed_accum_t)v_threshold[w];
                    else
                        next_voltage = (ed_accum_t)v_reset[w];
                }

                const bool remains_active =
                    ed_abs_accum(next_drive) > noise_threshold
                    || ed_abs_accum(next_voltage - (ed_accum_t)v_leak[w])
                        > noise_threshold;
                tick_survivor[slot] = remains_active;
                if (remains_active) {
                    current_drive[neuron] = (ed_current_t)next_drive;
                    v_mem[neuron] = (ed_voltage_t)next_voltage;
                    active_flag[neuron] = true;
                } else {
                    current_drive[neuron] = 0;
                    v_mem[neuron] = (ed_voltage_t)v_leak[w];
                    active_flag[neuron] = false;
                }
            }
            // The event stream is scalar, so serialize spikes after the
            // scalar state update and before compacting active_ids.
            for (unsigned int slot = 0; slot < old_active_count; ++slot) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=TOTAL_NEURONS
                #pragma HLS PIPELINE II=1
                if (tick_spike[slot]) {
                    const unsigned int neuron = active_ids[slot];
                    const unsigned int w = neuron % (unsigned int)IN_W;
                    const unsigned int spatial = neuron / (unsigned int)IN_W;
                    const unsigned int h = spatial % (unsigned int)IN_H;
                    const unsigned int c = spatial / (unsigned int)IN_H;
                    ed_spike_t spike = event;
                    spike.type = ED_TYPE_SPIKE;
                    spike.amplitude = (ed_current_t)1;
                    spike.channel_idx = c;
                    spike.height_idx = h;
                    spike.width_idx = w;
                    out.write(spike);
                }
            }
            // Stable prefix/compaction preserves active-list order.
            for (unsigned int slot = 0; slot < old_active_count; ++slot) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=TOTAL_NEURONS
                #pragma HLS PIPELINE II=1
                if (tick_survivor[slot])
                    next_active_ids[next_active_count++] = active_ids[slot];
            }
            ed_active_compact:
            for (unsigned int slot = 0; slot < next_active_count; ++slot) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=TOTAL_NEURONS
                #pragma HLS PIPELINE II=1
                active_ids[slot] = next_active_ids[slot];
            }
            active_count = next_active_count;
            out.write(event);
            finished = true;

            if (event.type == ED_TYPE_END_SAMPLE) {
                active_count = 0;
                ed_active_sample_reset:
                for (int neuron = 0; neuron < TOTAL_NEURONS; ++neuron) {
                    const int w = (IN_C == 1 && IN_H == 1)
                        ? neuron : neuron % IN_W;
                    v_mem[neuron] = v_reset[w];
                    current_drive[neuron] = 0;
                    pending_drive[neuron] = 0;
                    active_ids[neuron] = 0;
                    next_active_ids[neuron] = 0;
                    const bool starts_active = ed_abs_accum(
                        (ed_accum_t)v_reset[w] - (ed_accum_t)v_leak[w]
                    ) > noise_threshold;
                    active_flag[neuron] = starts_active;
                    if (starts_active)
                        active_ids[active_count++] = neuron;
                }
            }
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
// CONV2D EVENT-DRIVEN
// =============================================================

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int D_H, int D_W, int GROUPS, bool HAS_BIAS,
    int C_IN, int H_IN, int W_IN, int C_OUT,
    typename params_type>
void ed_conv2d_impl(
    hls::stream<ed_spike_t>& in,
    hls::stream<ed_spike_t>& out,
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type *bias)
{
        const int H_OUT = (H_IN + 2 * P_H - (D_H * (K_H - 1) + 1)) / S_H + 1;
    const int W_OUT = (W_IN + 2 * P_W - (D_W * (K_W - 1) + 1)) / S_W + 1;
    const int C_IN_GROUP = C_IN / GROUPS;
    const int C_OUT_GROUP = C_OUT / GROUPS;
    ed_accum_t accum[C_OUT][H_OUT][W_OUT];

    ed_conv_init_c:
    for (int c = 0; c < C_OUT; ++c)
        for (int h = 0; h < H_OUT; ++h)
            for (int w = 0; w < W_OUT; ++w)
                accum[c][h][w] = HAS_BIAS ? (ed_accum_t)bias[c] : (ed_accum_t)0;

    ed_spike_t end_token;
    bool done = false;
    ed_conv_read:
    while (!done) {
        ed_spike_t spike = in.read();
        if (spike.type == ED_TYPE_SPIKE) {
            const int in_c = spike.channel_idx;
            const int in_h = spike.height_idx;
            const int in_w = spike.width_idx;
            if (in_c >= 0 && in_c < C_IN && in_h >= 0 && in_h < H_IN && in_w >= 0 && in_w < W_IN) {
                const int group = in_c / C_IN_GROUP;
                const int out_begin = group * C_OUT_GROUP;
                const int weight_c = in_c - group * C_IN_GROUP;
                for (int kh = 0; kh < K_H; ++kh) {
                    for (int kw = 0; kw < K_W; ++kw) {
                        const int numerator_h = in_h + P_H - kh * D_H;
                        const int numerator_w = in_w + P_W - kw * D_W;
                        if (numerator_h >= 0 && numerator_w >= 0 &&
                            numerator_h % S_H == 0 && numerator_w % S_W == 0) {
                            const int out_h = numerator_h / S_H;
                            const int out_w = numerator_w / S_W;
                            if (out_h < H_OUT && out_w < W_OUT) {
                                for (int out_c_offset = 0;
                                     out_c_offset < C_OUT_GROUP; ++out_c_offset) {
                                    #pragma HLS PIPELINE II=1
                                    const int out_c = out_begin + out_c_offset;
                                    accum[out_c][out_h][out_w] +=
                                        (ed_accum_t)spike.amplitude *
                                        (ed_accum_t)weights[out_c][weight_c][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        } else if (spike.type == ED_TYPE_END_STEP || spike.type == ED_TYPE_END_SAMPLE) {
            end_token = spike;
            done = true;
        }
    }

    ed_conv_write_c:
    for (int c = 0; c < C_OUT; ++c) {
        for (int h = 0; h < H_OUT; ++h) {
            for (int w = 0; w < W_OUT; ++w) {
                if (accum[c][h][w] != (ed_accum_t)0) {
                    ed_spike_t spike = end_token;
                    spike.type = ED_TYPE_SPIKE;
                    spike.amplitude = (ed_current_t)accum[c][h][w];
                    spike.channel_idx = c;
                    spike.height_idx = h;
                    spike.width_idx = w;
                    out.write(spike);
                }
            }
        }
    }
    out.write(end_token);
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int D_H, int D_W, int GROUPS,
    int C_IN, int H_IN, int W_IN, int C_OUT,
    typename params_type>
void Conv2d(
    hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out,
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type (&bias)[C_OUT])
{
    ed_conv2d_impl<K_H, K_W, S_H, S_W, P_H, P_W, D_H, D_W,
                   GROUPS, true, C_IN, H_IN, W_IN, C_OUT>(in, out, weights, bias);
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int D_H, int D_W, int GROUPS,
    int C_IN, int H_IN, int W_IN, int C_OUT,
    typename params_type>
void Conv2d(
    hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out,
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W])
{
    ed_conv2d_impl<K_H, K_W, S_H, S_W, P_H, P_W, D_H, D_W,
                   GROUPS, false, C_IN, H_IN, W_IN, C_OUT>(
        in, out, weights, (const params_type *)0);
}

// =============================================================
// SUMPOOL2D EVENT-DRIVEN
// =============================================================

template<int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
         int C, int H_IN, int W_IN>
void SumPool2d(hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out)
{
        const int H_OUT = (H_IN + 2 * P_H - K_H) / S_H + 1;
    const int W_OUT = (W_IN + 2 * P_W - K_W) / S_W + 1;
    ed_accum_t accum[C][H_OUT][W_OUT] = {};
    ed_spike_t end_token;
    bool done = false;

    while (!done) {
        ed_spike_t spike = in.read();
        if (spike.type == ED_TYPE_SPIKE) {
            const int c = spike.channel_idx;
            const int in_h = spike.height_idx;
            const int in_w = spike.width_idx;
            if (c >= 0 && c < C && in_h >= 0 && in_h < H_IN && in_w >= 0 && in_w < W_IN) {
                for (int kernel_index = 0; kernel_index < K_H * K_W;
                     ++kernel_index) {
                    #pragma HLS PIPELINE II=1
                    const int kh = kernel_index / K_W;
                    const int kw = kernel_index % K_W;
                    const int numerator_h = in_h + P_H - kh;
                    const int numerator_w = in_w + P_W - kw;
                    if (numerator_h >= 0 && numerator_w >= 0 &&
                        numerator_h % S_H == 0 && numerator_w % S_W == 0) {
                        const int out_h = numerator_h / S_H;
                        const int out_w = numerator_w / S_W;
                        if (out_h < H_OUT && out_w < W_OUT)
                            accum[c][out_h][out_w] += spike.amplitude;
                    }
                }
            }
        } else if (spike.type == ED_TYPE_END_STEP || spike.type == ED_TYPE_END_SAMPLE) {
            end_token = spike;
            done = true;
        }
    }

    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H_OUT; ++h)
            for (int w = 0; w < W_OUT; ++w)
                if (accum[c][h][w] != (ed_accum_t)0) {
                    ed_spike_t spike = end_token;
                    spike.type = ED_TYPE_SPIKE;
                    spike.amplitude = (ed_current_t)accum[c][h][w];
                    spike.channel_idx = c;
                    spike.height_idx = h;
                    spike.width_idx = w;
                    out.write(spike);
                }
    out.write(end_token);
}

// =============================================================
// IF E LIF EVENT-DRIVEN
// =============================================================

// Advance a plain LIF neuron only when an input event addresses it.  The
// input amplitude keeps the discrete model's dt/tau scaling, while the state
// accumulated since the previous event is evolved analytically.  This makes
// idle neurons consume no update work and prevents a marker/tick from
// producing a spike by itself.
inline bool ed_update_lif_on_event_pwl(
    ed_voltage_t* state,
    ed_time_step_t* last_update_time,
    const ed_spike_t& input_event,
    ed_accum_t tau,
    ed_accum_t r,
    ed_accum_t v_leak,
    ed_accum_t threshold,
    ed_accum_t v_reset,
    ed_accum_t input_dt)
{
    #pragma HLS INLINE off
    const ed_accum_t min_tau = (ed_accum_t)(1.0 / 65536.0);
    const ed_accum_t safe_tau = tau > (ed_accum_t)0 ? tau : min_tau;
    const ed_accum_t elapsed =
        (ed_accum_t)input_event.timestamp - (ed_accum_t)(*last_update_time);

    if (elapsed > (ed_accum_t)0) {
        const ed_accum_t decay = (ed_accum_t)ed_exp_pwl_lut_decay_fp(
            -(elapsed / safe_tau));
        *state = (ed_voltage_t)(
            v_leak + ((ed_accum_t)(*state) - v_leak) * decay);
        *last_update_time = input_event.timestamp;
    }

    const ed_accum_t event_delta = (input_dt / safe_tau) * r
        * (ed_accum_t)input_event.amplitude;
    const ed_accum_t next_state = (ed_accum_t)(*state) + event_delta;
    if (next_state >= threshold) {
        *state = (ed_voltage_t)v_reset;
        return true;
    }

    *state = (ed_voltage_t)next_state;
    return false;
}

template<int C, int H, int W, int INSTANCE_ID = 0,
         typename params_type>
void IF(
    hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out,
    const params_type (&r)[C][H][W],
    const params_type (&threshold)[C][H][W],
    const params_type (&v_reset)[C][H][W], bool reset_potentials = false)
{
        const int TOTAL = C * H * W;
    static ed_voltage_t state[TOTAL] = {};
    ed_accum_t input[TOTAL] = {};
    bool fired[TOTAL] = {};
    ed_spike_t end_token;
    bool done = false;
    while (!done) {
        ed_spike_t spike = in.read();
        if (spike.type == ED_TYPE_SPIKE) {
            const int c = spike.channel_idx, h = spike.height_idx, w = spike.width_idx;
            if (c >= 0 && c < C && h >= 0 && h < H && w >= 0 && w < W) {
                const int index = (c * H + h) * W + w;
                input[index] += spike.amplitude;
            }
        } else if (spike.type == ED_TYPE_END_STEP || spike.type == ED_TYPE_END_SAMPLE) {
            end_token = spike;
            done = true;
        }
    }
    for (int index = 0; index < TOTAL; ++index) {
        #pragma HLS PIPELINE II=1
        const int w = index % W;
        const int spatial = index / W;
        const int h = spatial % H;
        const int c = spatial / H;
        if (reset_potentials)
            state[index] = (ed_voltage_t)v_reset[c][h][w];
        state[index] += input[index] * (ed_accum_t)r[c][h][w];
        if (state[index] >= (ed_voltage_t)threshold[c][h][w]) {
            fired[index] = true;
            state[index] = (ed_voltage_t)v_reset[c][h][w];
        }
        if (end_token.type == ED_TYPE_END_SAMPLE)
            state[index] = (ed_voltage_t)v_reset[c][h][w];
    }
    for (int index = 0; index < TOTAL; ++index) {
        #pragma HLS PIPELINE II=1
        if (fired[index]) {
            const int w = index % W;
            const int spatial = index / W;
            const int h = spatial % H;
            const int c = spatial / H;
            ed_spike_t spike = end_token;
            spike.type = ED_TYPE_SPIKE;
            spike.amplitude = 1;
            spike.channel_idx = c;
            spike.height_idx = h;
            spike.width_idx = w;
            out.write(spike);
        }
    }
    out.write(end_token);
}

template<int C, int H, int W, int INSTANCE_ID = 0,
         typename tau_type, typename params_type>
void LIF(
    hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out,
    const tau_type (&tau)[C][H][W], const params_type (&r)[C][H][W],
    const params_type (&v_leak)[C][H][W],
    const params_type (&threshold)[C][H][W],
    const params_type (&v_reset)[C][H][W], tau_type dt,
    bool reset_potentials = false)
{
    static ed_voltage_t state[C][H][W] = {};
    static ed_time_step_t last_update_time[C][H][W] = {};
    if (reset_potentials) {
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    state[c][h][w] = (ed_voltage_t)v_reset[c][h][w];
                    last_update_time[c][h][w] = (ed_time_step_t)0;
                }
    }
    bool done = false;
    while (!done) {
        ed_spike_t spike = in.read();
        if (spike.type == ED_TYPE_SPIKE) {
            const int c = spike.channel_idx, h = spike.height_idx, w = spike.width_idx;
            if (c >= 0 && c < C && h >= 0 && h < H && w >= 0 && w < W) {
                if (ed_update_lif_on_event_pwl(
                        &state[c][h][w], &last_update_time[c][h][w], spike,
                        (ed_accum_t)tau[c][h][w], (ed_accum_t)r[c][h][w],
                        (ed_accum_t)v_leak[c][h][w],
                        (ed_accum_t)threshold[c][h][w],
                        (ed_accum_t)v_reset[c][h][w], (ed_accum_t)dt)) {
                    spike.amplitude = 1;
                    out.write(spike);
                }
            }
        } else if (spike.type == ED_TYPE_END_STEP || spike.type == ED_TYPE_END_SAMPLE) {
            if (spike.type == ED_TYPE_END_SAMPLE) {
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            state[c][h][w] = (ed_voltage_t)v_reset[c][h][w];
                            last_update_time[c][h][w] = (ed_time_step_t)0;
                        }
            }
            out.write(spike);
            done = true;
        }
    }
}

template<int N, int INSTANCE_ID = 0, typename params_type>
void IF(
    hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out,
    const params_type (&r)[N], const params_type (&threshold)[N],
    const params_type (&v_reset)[N], bool reset_potentials = false)
{
        static ed_voltage_t state[N] = {};
    ed_accum_t input[N] = {};
    bool fired[N] = {};
    ed_spike_t end_token;
    bool done = false;
    while (!done) {
        ed_spike_t spike = in.read();
        if (spike.type == ED_TYPE_SPIKE) {
            const int index = spike.width_idx;
            if (index >= 0 && index < N) input[index] += spike.amplitude;
        } else if (spike.type == ED_TYPE_END_STEP || spike.type == ED_TYPE_END_SAMPLE) {
            end_token = spike;
            done = true;
        }
    }
    for (int index = 0; index < N; ++index) {
        #pragma HLS PIPELINE II=1
        if (reset_potentials) state[index] = (ed_voltage_t)v_reset[index];
        state[index] += input[index] * (ed_accum_t)r[index];
        if (state[index] >= (ed_voltage_t)threshold[index]) {
            fired[index] = true;
            state[index] = (ed_voltage_t)v_reset[index];
        }
        if (end_token.type == ED_TYPE_END_SAMPLE)
            state[index] = (ed_voltage_t)v_reset[index];
    }
    for (int index = 0; index < N; ++index) {
        #pragma HLS PIPELINE II=1
        if (fired[index]) {
            ed_spike_t spike = end_token;
            spike.type = ED_TYPE_SPIKE;
            spike.amplitude = 1;
            spike.channel_idx = 0;
            spike.height_idx = 0;
            spike.width_idx = index;
            out.write(spike);
        }
    }
    out.write(end_token);
}

template<int N, int INSTANCE_ID = 0, typename tau_type, typename params_type>
void LIF(
    hls::stream<ed_spike_t>& in, hls::stream<ed_spike_t>& out,
    const tau_type (&tau)[N], const params_type (&r)[N],
    const params_type (&v_leak)[N], const params_type (&threshold)[N],
    const params_type (&v_reset)[N], tau_type dt,
    bool reset_potentials = false)
{
    static ed_voltage_t state[N] = {};
    static ed_time_step_t last_update_time[N] = {};
    if (reset_potentials) {
        for (int index = 0; index < N; ++index) {
            state[index] = (ed_voltage_t)v_reset[index];
            last_update_time[index] = (ed_time_step_t)0;
        }
    }
    bool done = false;
    while (!done) {
        ed_spike_t spike = in.read();
        if (spike.type == ED_TYPE_SPIKE) {
            const int index = spike.width_idx;
            if (index >= 0 && index < N) {
                if (ed_update_lif_on_event_pwl(
                        &state[index], &last_update_time[index], spike,
                        (ed_accum_t)tau[index], (ed_accum_t)r[index],
                        (ed_accum_t)v_leak[index],
                        (ed_accum_t)threshold[index],
                        (ed_accum_t)v_reset[index], (ed_accum_t)dt)) {
                    spike.amplitude = 1;
                    spike.channel_idx = 0;
                    spike.height_idx = 0;
                    out.write(spike);
                }
            }
        } else if (spike.type == ED_TYPE_END_STEP || spike.type == ED_TYPE_END_SAMPLE) {
            if (spike.type == ED_TYPE_END_SAMPLE) {
                for (int index = 0; index < N; ++index) {
                    state[index] = (ed_voltage_t)v_reset[index];
                    last_update_time[index] = (ed_time_step_t)0;
                }
            }
            out.write(spike);
            done = true;
        }
    }
}
