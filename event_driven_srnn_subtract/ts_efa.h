#ifndef TS_EFA_H
#define TS_EFA_H

#include <stdint.h>
#include "types.h"

// Precision settings for fixed-point math
#define FRACTIONAL_BITS 8
#define ONE_FP (1 << FRACTIONAL_BITS)
#define DECAY_FRACTIONAL_BITS 16
#define DECAY_ONE_FP (1u << DECAY_FRACTIONAL_BITS)
#define TS_EFA_LUT_SIZE 256

static const uint32_t TEMP_LUT[TS_EFA_LUT_SIZE] = {
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

static const uint32_t SCAL_LUT[TS_EFA_LUT_SIZE] = {
    65536, 32768, 16384, 8192, 4096, 2048, 1024, 512,
    256, 128, 64, 32, 16, 8, 4, 2,
    1, 1
};

/**
 * @brief Template Scaling Exponential Function Approximation (TS-EFA)
 * 
 * This function efficiently approximates the exponential decay exp(-dt / tau)
 * typically used in event-driven Spiking Neural Networks. It reformulates the
 * equation into base 2: 2^(-dt / (tau * ln(2))) and multiplies a template LUT
 * by a scaling LUT, matching the TS-EFA EXP_FIX structure.
 * 
 * @param dt_scaled The time difference already scaled by (1 / (tau * ln(2)))
 *                  in Q8 fixed-point format.
 * @return The decay multiplier in Q16 fixed-point format.
 */
uint32_t ts_efa_compute_decay(uint32_t dt_scaled) {
    #pragma HLS INLINE off
    uint32_t dt_int = dt_scaled >> FRACTIONAL_BITS;            
    uint32_t dt_frac = dt_scaled & (ONE_FP - 1); 

    if (dt_int >= TS_EFA_LUT_SIZE) {
        return 0;
    }

    uint64_t product = (uint64_t)TEMP_LUT[dt_frac] * (uint64_t)SCAL_LUT[dt_int];
    return (uint32_t)((product + (DECAY_ONE_FP >> 1)) >> DECAY_FRACTIONAL_BITS);
}

/**
 * @brief Adapter for CUBA LIF fixed-point state update.
 *
 * The neuron update passes the exponential argument directly as -dt / tau.
 * TS-EFA uses the positive base-2 domain, so this converts:
 *   exp(-dt/tau) = 2^(-dt/(tau*ln(2)))
 */
inline decay_t ts_efa_compute_decay_fp(accum_t exp_arg) {
    #pragma HLS INLINE off
    const accum_t INV_LN2 = (accum_t)1.44269504088896340736;

    if (exp_arg >= (accum_t)0) {
        return (decay_t)1;
    }

    accum_t scaled = (-exp_arg) * INV_LN2;
    uint32_t scaled_fp = (uint32_t)(scaled * (accum_t)ONE_FP);
    uint32_t decay_fp = ts_efa_compute_decay(scaled_fp);

    ap_ufixed<17, 1> decay_value = 0;
    decay_value.range(16, 0) = (ap_uint<17>)decay_fp;
    return (decay_t)decay_value;
}

#endif // TS_EFA_H
