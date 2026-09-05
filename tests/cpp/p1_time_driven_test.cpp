#include <cassert>
#include <cmath>

#include "neuro_hls_functions/time_driven.h"
// Legacy include must remain source-compatible during the rename.
#include "neuro_hls_functions/dense.h"

static bool close(float left, float right) {
    return std::fabs(left - right) < 1e-5f;
}

int main() {
    {
        float input[1][5] = {{1, 2, 3, 4, 5}};
        float weights[1][1][2] = {{{1, 1}}};
        float bias[1] = {0.5f};
        float output[1][3] = {};
        Conv1d<2, 1, 0, 2, 1>(input, output, weights, bias);
        assert(close(output[0][0], 4.5f));
        assert(close(output[0][1], 6.5f));
        assert(close(output[0][2], 8.5f));
    }

    {
        float input[2][3] = {{1, 2, 3}, {4, 5, 6}};
        float weights[2][1][1] = {{{2}}, {{3}}};
        float output[2][3] = {};
        Conv1d<1, 1, 0, 1, 2>(input, output, weights);
        assert(close(output[0][2], 6.0f));
        assert(close(output[1][0], 12.0f));
    }

    {
        float input[1][2][2] = {{{1, 2}, {3, 4}}};
        float output[1][3][3] = {};
        AvgPool2d<2, 2, 1, 1, 1, 1>(input, output);
        assert(close(output[0][0][0], 0.25f));
        assert(close(output[0][1][1], 2.5f));
        assert(close(output[0][2][2], 1.0f));
    }

    {
        float input[1][2][2] = {{{1, 2}, {3, 4}}};
        float weights[1][1][1][1] = {{{{2}}}};
        float output[1][2][2] = {};
        Conv2d<1, 1, 1, 1, 0, 0, 1, 1, 1>(input, output, weights);
        assert(close(output[0][0][1], 4.0f));
        assert(close(output[0][1][1], 8.0f));
    }

    {
        float input_1d[3] = {1, 2, 3};
        float scale_1d[3] = {2, 3, 4};
        float output_1d[3] = {};
        Scale(input_1d, output_1d, scale_1d);
        assert(close(output_1d[2], 12.0f));

        float input_2d[2][2] = {{1, 2}, {3, 4}};
        float scale_2d[2][2] = {{2, 2}, {2, 2}};
        float output_2d[2][2] = {};
        Scale(input_2d, output_2d, scale_2d);
        assert(close(output_2d[1][0], 6.0f));

        float input_3d[1][1][2] = {{{2, 4}}};
        float scale_3d[1][1][2] = {{{0.5f, 0.25f}}};
        float output_3d[1][1][2] = {};
        Scale(input_3d, output_3d, scale_3d);
        assert(close(output_3d[0][0][0], 1.0f));
        assert(close(output_3d[0][0][1], 1.0f));
    }

    return 0;
}
