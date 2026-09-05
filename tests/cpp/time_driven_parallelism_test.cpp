#include <cassert>
#include <cmath>

#include "neuro_hls_functions/time_driven.h"


static bool close(float left, float right) {
    return std::fabs(left - right) < 1e-5f;
}


template<int N>
static void assert_same(const float (&serial)[N], const float (&parallel)[N]) {
    for (int i = 0; i < N; ++i) {
        assert(close(serial[i], parallel[i]));
    }
}


template<int N>
static void assert_same(const double (&serial)[N],
                        const double (&parallel)[N]) {
    for (int i = 0; i < N; ++i) {
        assert(std::fabs(serial[i] - parallel[i]) < 1e-9);
    }
}


template<int N>
static void assert_same(const bit_t (&serial)[N], const bit_t (&parallel)[N]) {
    for (int i = 0; i < N; ++i) {
        assert(static_cast<unsigned int>(serial[i]) ==
               static_cast<unsigned int>(parallel[i]));
    }
}


template<int H, int W>
static void assert_same(const float (&serial)[H][W],
                        const float (&parallel)[H][W]) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            assert(close(serial[h][w], parallel[h][w]));
        }
    }
}


template<int H, int W>
static void assert_same(const double (&serial)[H][W],
                        const double (&parallel)[H][W]) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            assert(std::fabs(serial[h][w] - parallel[h][w]) < 1e-9);
        }
    }
}


template<int H, int W>
static void assert_same(const bit_t (&serial)[H][W],
                        const bit_t (&parallel)[H][W]) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            assert(static_cast<unsigned int>(serial[h][w]) ==
                   static_cast<unsigned int>(parallel[h][w]));
        }
    }
}


template<int C, int H, int W>
static void assert_same(const float (&serial)[C][H][W],
                        const float (&parallel)[C][H][W]) {
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                assert(close(serial[c][h][w], parallel[c][h][w]));
            }
        }
    }
}


template<int C, int H, int W>
static void assert_same(const bit_t (&serial)[C][H][W],
                        const bit_t (&parallel)[C][H][W]) {
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                assert(static_cast<unsigned int>(serial[c][h][w]) ==
                       static_cast<unsigned int>(parallel[c][h][w]));
            }
        }
    }
}


int main() {
    // Elementwise primitives: five values split over three lanes exercises
    // the guarded tail tile.
    {
        float serial[5] = {1, 2, 3, 4, 5};
        float parallel[5] = {1, 2, 3, 4, 5};
        float other[5] = {10, 20, 30, 40, 50};
        Merge<1>(serial, other);
        Merge<3>(parallel, other);
        assert_same(serial, parallel);
    }

    {
        float input[5] = {1, 2, 3, 4, 5};
        float serial[5] = {};
        float parallel[5] = {};
        Flatten<5, 1>(input, serial);
        Flatten<5, 3>(input, parallel);
        assert_same(serial, parallel);
    }

    {
        float input[2][3] = {{1, 2, 3}, {4, 5, 6}};
        double serial[6] = {};
        double parallel[6] = {};
        Flatten<2, 3, 1>(input, serial);
        Flatten<2, 3, 4>(input, parallel);
        assert_same(serial, parallel);
    }

    {
        float input[1][1][5] = {{{1, 2, 3, 4, 5}}};
        float serial[5] = {};
        float parallel[5] = {};
        Flatten<1, 1, 5, 1>(input, serial);
        Flatten<1, 1, 5, 3>(input, parallel);
        assert_same(serial, parallel);
    }

    {
        float input[1][1][5] = {{{1, 2, 3, 4, 5}}};
        float factors[1][1][5] = {{{2, -1, 0.5f, 3, 4}}};
        float serial[1][1][5] = {};
        float parallel[1][1][5] = {};
        Scale<1, 1, 5, 1>(input, serial, factors);
        Scale<1, 1, 5, 3>(input, parallel, factors);
        for (int i = 0; i < 5; ++i) {
            assert(close(serial[0][0][i], parallel[0][0][i]));
        }
    }

    // Dense reuse uses a flat domain of 15 MACs.  Eight processing elements
    // exercise the guarded tail of the second reuse group.
    {
        float input[5] = {1, 2, 3, 4, 5};
        float weights[3][5] = {
            {1, 2, 3, 4, 5},
            {-1, 0, 1, 0, -1},
            {2, 2, 2, 2, 2},
        };
        float serial[3] = {};
        float parallel[3] = {};
        LinearReuse<1>(input, serial, weights);
        LinearReuse<8>(input, parallel, weights);
        assert_same(serial, parallel);
    }

    {
        float input[5] = {1, 2, 3, 4, 5};
        float weights[3][5] = {
            {1, 2, 3, 4, 5},
            {-1, 0, 1, 0, -1},
            {2, 2, 2, 2, 2},
        };
        float bias[3] = {0.5f, -2, 4};
        float serial[3] = {};
        float parallel[3] = {};
        AffineReuse<1>(input, serial, weights, bias);
        AffineReuse<8>(input, parallel, weights, bias);
        assert_same(serial, parallel);
    }

    // Conv1d compares a serial MAC schedule with seven processing elements.
    {
        float input[2][5] = {
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
        };
        float weights[2][2][3] = {
            {{1, 0, -1}, {2, 1, 0}},
            {{-1, 2, 1}, {0, 1, 2}},
        };
        float bias[2] = {0.25f, -0.5f};
        float serial[2][5] = {};
        float parallel[2][5] = {};
        Conv1dReuse<3, 1, 1, 1, 1, 1>(input, serial, weights, bias);
        Conv1dReuse<3, 1, 1, 1, 1, 7>(input, parallel, weights, bias);
        for (int c = 0; c < 2; ++c) {
            for (int w = 0; w < 5; ++w) {
                assert(close(serial[c][w], parallel[c][w]));
            }
        }
    }

    // Conv2d uses a flat output-by-kernel MAC schedule.
    {
        float input[2][4][4] = {
            {{1, 2, 3, 4}, {5, 6, 7, 8},
             {9, 10, 11, 12}, {13, 14, 15, 16}},
            {{16, 15, 14, 13}, {12, 11, 10, 9},
             {8, 7, 6, 5}, {4, 3, 2, 1}},
        };
        float weights[2][2][2][2] = {
            {{{1, 0}, {0, -1}}, {{2, 1}, {1, 0}}},
            {{{-1, 2}, {0, 1}}, {{1, 0}, {2, -1}}},
        };
        float bias[2] = {0.5f, -1};
        float serial[2][3][3] = {};
        float parallel[2][3][3] = {};
        Conv2dReuse<2, 2, 1, 1, 0, 0, 1, 1, 1, 1>(
            input, serial, weights, bias
        );
        Conv2dReuse<2, 2, 1, 1, 0, 0, 1, 1, 1, 7>(
            input, parallel, weights, bias
        );
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 3; ++w) {
                    assert(close(serial[c][h][w], parallel[c][h][w]));
                }
            }
        }
    }

    // Pooling uses the same flat reuse schedule and a non-divisor PE count.
    {
        float input[2][4][5] = {
            {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10},
             {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}},
            {{20, 19, 18, 17, 16}, {15, 14, 13, 12, 11},
             {10, 9, 8, 7, 6}, {5, 4, 3, 2, 1}},
        };
        float serial[2][3][2] = {};
        float parallel[2][3][2] = {};
        SumPool2dReuse<2, 3, 1, 2, 0, 0, 1>(input, serial);
        SumPool2dReuse<2, 3, 1, 2, 0, 0, 7>(input, parallel);
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 2; ++w) {
                    assert(close(serial[c][h][w], parallel[c][h][w]));
                }
            }
        }
    }

    {
        float input[2][4][5] = {
            {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10},
             {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}},
            {{20, 19, 18, 17, 16}, {15, 14, 13, 12, 11},
             {10, 9, 8, 7, 6}, {5, 4, 3, 2, 1}},
        };
        float serial[2][3][2] = {};
        float parallel[2][3][2] = {};
        AvgPool2dReuse<2, 3, 1, 2, 0, 0, 1>(input, serial);
        AvgPool2dReuse<2, 3, 1, 2, 0, 0, 7>(input, parallel);
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 2; ++w) {
                    assert(close(serial[c][h][w], parallel[c][h][w]));
                }
            }
        }
    }

    // Neuron wrappers are stateful; compare two steps so the test covers
    // both initialization/reset and persistent state updates.
    {
        float input[5] = {0.25f, 1, 2, 3, 4};
        float resistance[5] = {1, 1, 0.5f, 1, 2};
        float threshold[5] = {1, 1, 1, 2, 3};
        float reset[5] = {0, -1, 0, 0.5f, 0};
        float serial_state[5] = {};
        float parallel_state[5] = {};
        bit_t serial_spikes[5] = {};
        bit_t parallel_spikes[5] = {};
        IF<5, 1>(input, serial_spikes, serial_state,
                 resistance, threshold, reset, true);
        IF<5, 3>(input, parallel_spikes, parallel_state,
                 resistance, threshold, reset, true);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
        IF<5, 1>(input, serial_spikes, serial_state,
                 resistance, threshold, reset, false);
        IF<5, 3>(input, parallel_spikes, parallel_state,
                 resistance, threshold, reset, false);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[2][3] = {{0.25f, 1, 2}, {3, 4, 0.5f}};
        float resistance[2][3] = {{1, 1, 0.5f}, {1, 2, 1}};
        float threshold[2][3] = {{1, 1, 1}, {2, 3, 1}};
        float reset[2][3] = {{0, -1, 0}, {0.5f, 0, -0.5f}};
        double serial_state[2][3] = {};
        double parallel_state[2][3] = {};
        bit_t serial_spikes[2][3] = {};
        bit_t parallel_spikes[2][3] = {};
        IF<2, 3, 1>(input, serial_spikes, serial_state,
                    resistance, threshold, reset, true);
        IF<2, 3, 4>(input, parallel_spikes, parallel_state,
                    resistance, threshold, reset, true);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
        IF<2, 3, 1>(input, serial_spikes, serial_state,
                    resistance, threshold, reset, false);
        IF<2, 3, 4>(input, parallel_spikes, parallel_state,
                    resistance, threshold, reset, false);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[5] = {0.25f, 1, 2, 3, 4};
        float tau[5] = {0.001f, 0.002f, 0.001f, 0.002f, 0.001f};
        float resistance[5] = {1, 1, 0.5f, 1, 2};
        float leak[5] = {0, 0, 0.25f, 0, -0.5f};
        float threshold[5] = {1, 1, 1, 2, 3};
        float reset[5] = {0, -1, 0, 0.5f, 0};
        float serial_state[5] = {};
        float parallel_state[5] = {};
        bit_t serial_spikes[5] = {};
        bit_t parallel_spikes[5] = {};
        LIF<5, 1>(input, serial_spikes, serial_state, tau, resistance,
                  leak, threshold, reset, true);
        LIF<5, 3>(input, parallel_spikes, parallel_state, tau, resistance,
                  leak, threshold, reset, true);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
        LIF<5, 1>(input, serial_spikes, serial_state, tau, resistance,
                  leak, threshold, reset, false);
        LIF<5, 3>(input, parallel_spikes, parallel_state, tau, resistance,
                  leak, threshold, reset, false);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[2][3] = {{0.25f, 1, 2}, {3, 4, 0.5f}};
        float tau[2][3] = {
            {0.001f, 0.002f, 0.001f},
            {0.002f, 0.001f, 0.002f},
        };
        float resistance[2][3] = {{1, 1, 0.5f}, {1, 2, 1}};
        float leak[2][3] = {{0, 0, 0.25f}, {0, -0.5f, 0}};
        float threshold[2][3] = {{1, 1, 1}, {2, 3, 1}};
        float reset[2][3] = {{0, -1, 0}, {0.5f, 0, -0.5f}};
        double serial_state[2][3] = {};
        double parallel_state[2][3] = {};
        bit_t serial_spikes[2][3] = {};
        bit_t parallel_spikes[2][3] = {};
        LIF<2, 3, 1>(input, serial_spikes, serial_state, tau, resistance,
                     leak, threshold, reset, true);
        LIF<2, 3, 4>(input, parallel_spikes, parallel_state, tau, resistance,
                     leak, threshold, reset, true);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
        LIF<2, 3, 1>(input, serial_spikes, serial_state, tau, resistance,
                     leak, threshold, reset, false);
        LIF<2, 3, 4>(input, parallel_spikes, parallel_state, tau, resistance,
                     leak, threshold, reset, false);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[1][2][3] = {{{0.25f, 1, 2}, {3, 4, 0.5f}}};
        float tau[1][2][3] = {{
            {0.001f, 0.002f, 0.001f},
            {0.002f, 0.001f, 0.002f},
        }};
        float resistance[1][2][3] = {{{1, 1, 0.5f}, {1, 2, 1}}};
        float leak[1][2][3] = {{{0, 0, 0.25f}, {0, -0.5f, 0}}};
        float threshold[1][2][3] = {{{1, 1, 1}, {2, 3, 1}}};
        float reset[1][2][3] = {{{0, -1, 0}, {0.5f, 0, -0.5f}}};
        float serial_state[1][2][3] = {};
        float parallel_state[1][2][3] = {};
        bit_t serial_spikes[1][2][3] = {};
        bit_t parallel_spikes[1][2][3] = {};
        LIF<1, 2, 3, 1>(input, serial_spikes, serial_state, tau, resistance,
                        leak, threshold, reset, true);
        LIF<1, 2, 3, 4>(input, parallel_spikes, parallel_state, tau,
                        resistance, leak, threshold, reset, true);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
        LIF<1, 2, 3, 1>(input, serial_spikes, serial_state, tau, resistance,
                        leak, threshold, reset, false);
        LIF<1, 2, 3, 4>(input, parallel_spikes, parallel_state, tau,
                        resistance, leak, threshold, reset, false);
        assert_same(serial_state, parallel_state);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[5] = {0.25f, 1, 2, 3, 4};
        float alpha[5] = {0.1f, 0.2f, 0.1f, 0.2f, 0.1f};
        float beta[5] = {0.2f, 0.1f, 0.2f, 0.1f, 0.2f};
        float resistance[5] = {1, 1, 0.5f, 1, 2};
        float leak[5] = {0, 0, 0.25f, 0, -0.5f};
        float threshold[5] = {1, 1, 1, 2, 3};
        float reset[5] = {0, -1, 0, 0.5f, 0};
        float input_weight[5] = {1, 1, 2, 0.5f, 1};
        float serial_current[5] = {};
        float parallel_current[5] = {};
        float serial_voltage[5] = {};
        float parallel_voltage[5] = {};
        bit_t serial_spikes[5] = {};
        bit_t parallel_spikes[5] = {};

        CubaLIF<float, 1>(
            input, serial_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, serial_current, serial_voltage, true, false
        );
        CubaLIF<float, 3>(
            input, parallel_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, parallel_current, parallel_voltage, true,
            false
        );
        assert_same(serial_current, parallel_current);
        assert_same(serial_voltage, parallel_voltage);
        assert_same(serial_spikes, parallel_spikes);

        CubaLIF<float, 1>(
            input, serial_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, serial_current, serial_voltage, false, false
        );
        CubaLIF<float, 3>(
            input, parallel_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, parallel_current, parallel_voltage, false,
            false
        );
        assert_same(serial_current, parallel_current);
        assert_same(serial_voltage, parallel_voltage);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[2][3] = {{0.25f, 1, 2}, {3, 4, 0.5f}};
        float alpha[2][3] = {{0.1f, 0.2f, 0.1f}, {0.2f, 0.1f, 0.2f}};
        float beta[2][3] = {{0.2f, 0.1f, 0.2f}, {0.1f, 0.2f, 0.1f}};
        float resistance[2][3] = {{1, 1, 0.5f}, {1, 2, 1}};
        float leak[2][3] = {{0, 0, 0.25f}, {0, -0.5f, 0}};
        float threshold[2][3] = {{1, 1, 1}, {2, 3, 1}};
        float reset[2][3] = {{0, -1, 0}, {0.5f, 0, -0.5f}};
        float input_weight[2][3] = {{1, 1, 2}, {0.5f, 1, 1.5f}};
        float serial_current[2][3] = {};
        float parallel_current[2][3] = {};
        float serial_voltage[2][3] = {};
        float parallel_voltage[2][3] = {};
        bit_t serial_spikes[2][3] = {};
        bit_t parallel_spikes[2][3] = {};

        CubaLIF<float, 1>(
            input, serial_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, serial_current, serial_voltage, true, false
        );
        CubaLIF<float, 4>(
            input, parallel_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, parallel_current, parallel_voltage, true,
            false
        );
        assert_same(serial_current, parallel_current);
        assert_same(serial_voltage, parallel_voltage);
        assert_same(serial_spikes, parallel_spikes);
        CubaLIF<float, 1>(
            input, serial_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, serial_current, serial_voltage, false, false
        );
        CubaLIF<float, 4>(
            input, parallel_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, parallel_current, parallel_voltage, false,
            false
        );
        assert_same(serial_current, parallel_current);
        assert_same(serial_voltage, parallel_voltage);
        assert_same(serial_spikes, parallel_spikes);
    }

    {
        float input[1][2][3] = {{{0.25f, 1, 2}, {3, 4, 0.5f}}};
        float alpha[1][2][3] = {{{0.1f, 0.2f, 0.1f},
                                  {0.2f, 0.1f, 0.2f}}};
        float beta[1][2][3] = {{{0.2f, 0.1f, 0.2f},
                                 {0.1f, 0.2f, 0.1f}}};
        float resistance[1][2][3] = {{{1, 1, 0.5f}, {1, 2, 1}}};
        float leak[1][2][3] = {{{0, 0, 0.25f}, {0, -0.5f, 0}}};
        float threshold[1][2][3] = {{{1, 1, 1}, {2, 3, 1}}};
        float reset[1][2][3] = {{{0, -1, 0}, {0.5f, 0, -0.5f}}};
        float input_weight[1][2][3] = {{{1, 1, 2}, {0.5f, 1, 1.5f}}};
        float serial_current[1][2][3] = {};
        float parallel_current[1][2][3] = {};
        float serial_voltage[1][2][3] = {};
        float parallel_voltage[1][2][3] = {};
        bit_t serial_spikes[1][2][3] = {};
        bit_t parallel_spikes[1][2][3] = {};

        CubaLIF<float, 1>(
            input, serial_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, serial_current, serial_voltage, true, false
        );
        CubaLIF<float, 4>(
            input, parallel_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, parallel_current, parallel_voltage, true,
            false
        );
        assert_same(serial_current, parallel_current);
        assert_same(serial_voltage, parallel_voltage);
        assert_same(serial_spikes, parallel_spikes);
        CubaLIF<float, 1>(
            input, serial_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, serial_current, serial_voltage, false, false
        );
        CubaLIF<float, 4>(
            input, parallel_spikes, alpha, beta, resistance, leak, threshold,
            reset, input_weight, parallel_current, parallel_voltage, false,
            false
        );
        assert_same(serial_current, parallel_current);
        assert_same(serial_voltage, parallel_voltage);
        assert_same(serial_spikes, parallel_spikes);
    }

    return 0;
}
