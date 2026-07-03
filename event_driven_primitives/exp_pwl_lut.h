#ifndef EXP_PWL_LUT_H
#define EXP_PWL_LUT_H

#include "types.h"

#define EXP_PWL_LUT_SEGMENTS 64

static const accum_t EXP_PWL_LUT_X_MIN = (accum_t)-8.0;
static const accum_t EXP_PWL_LUT_X_MAX = (accum_t)0.0;
static const accum_t EXP_PWL_LUT_STEP = (accum_t)0.125;
static const accum_t EXP_PWL_LUT_INV_STEP = (accum_t)8.0;

static const decay_t EXP_PWL_LUT_VALUE[EXP_PWL_LUT_SEGMENTS] = {
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

static const accum_t EXP_PWL_LUT_SLOPE[EXP_PWL_LUT_SEGMENTS] = {
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

/**
 * @brief Piecewise-linear LUT approximation of exp(x) for decay factors.
 *
 * The valid interpolation domain is x in [-8, 0], matching the usual
 * exp(-dt / tau) argument used by the CUBA LIF update. Inputs above 0
 * saturate to 1 and inputs below -8 saturate to 0.
 *
 * @param exp_arg Exponential argument, normally -dt / tau.
 * @return Approximation of exp(exp_arg) as a fixed-point decay multiplier.
 */
inline decay_t exp_pwl_lut_decay_fp(accum_t exp_arg) {
    #pragma HLS INLINE off
    if (exp_arg >= EXP_PWL_LUT_X_MAX) {
        return (decay_t)1;
    }

    if (exp_arg < EXP_PWL_LUT_X_MIN) {
        return (decay_t)0;
    }

    accum_t scaled_index = (exp_arg - EXP_PWL_LUT_X_MIN) * EXP_PWL_LUT_INV_STEP;
    int lut_idx = (int)scaled_index;

    if (lut_idx >= EXP_PWL_LUT_SEGMENTS) {
        lut_idx = EXP_PWL_LUT_SEGMENTS - 1;
    }

    accum_t segment_x = EXP_PWL_LUT_X_MIN + (accum_t)lut_idx * EXP_PWL_LUT_STEP;
    accum_t local_x = exp_arg - segment_x;
    accum_t result = (accum_t)EXP_PWL_LUT_VALUE[lut_idx]
                   + EXP_PWL_LUT_SLOPE[lut_idx] * local_x;

    if (result <= (accum_t)0) {
        return (decay_t)0;
    }

    if (result >= (accum_t)1) {
        return (decay_t)1;
    }

    return (decay_t)result;
}

#endif // EXP_PWL_LUT_H
