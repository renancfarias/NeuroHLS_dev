#ifndef _BIT_TYPE_H_
#define _BIT_TYPE_H_

#include "ap_int.h"

typedef ap_uint<1> bit_t;

#define THRESHOLD 1.0
#define DECAY 0.7

namespace layer {
    const ap_fixed<16, 8> decay = DECAY;
    const ap_fixed<16, 8> threshold = THRESHOLD;
}

#endif