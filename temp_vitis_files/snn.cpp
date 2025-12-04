#include <iostream>
#include <vector>
#include <ap_fixed.h>
#include <ap_int.h>
#include "layer.h"

// #include "n-mnist_weights.h"
// #include "ecg_ptbdb/weights.h"

#include "tobias_fashion_mnist/weights.h"

// void ann_like_sbesc_2025(bit_t input [NUM_INPUTS],
//                          bit_t output [num_neurons[1]],
//                          weight_t w1[num_neurons[0]][NUM_INPUTS],
//                          weight_t w2[num_neurons[1]][num_neurons[0]])
// {
//     #pragma HLS function_instantiate variable=input
//     bit_t output_first[num_neurons[0]];

//     ann_lif_layer<num_neurons[0], NUM_INPUTS, bit_t, 14, 128>(input, output_first, w1);
//     ann_lif_layer<num_neurons[1], num_neurons[0], bit_t, 16, 10>(output_first, output, w2);
// }

// void ann_like_lascas_2026(input_ecg_t input [NUM_INPUTS],
//                           bit_t output [num_neurons[1]],
//                           weight_t w1[num_neurons[0]][NUM_INPUTS],
//                           weight_t w2[num_neurons[1]][num_neurons[0]])
// {
//     #pragma HLS function_instantiate variable=input
//     bit_t output_first[num_neurons[0]];

//     // ann_lif_layer<num_neurons[0], NUM_INPUTS, input_ecg_t, 1, 1>(input, output_first, w1);
//     // ann_lif_layer<num_neurons[1], num_neurons[0], bit_t, 1, 1>(output_first, output, w2);

//     ann_lif_layer_debug<num_neurons[0], NUM_INPUTS, input_ecg_t, 1, 1, 10>(input, output_first, w1);
//     ann_lif_layer_debug<num_neurons[1], num_neurons[0], bit_t, 1, 1, 2>(output_first, output, w2);

//     // cout << "----------------" << endl;
// }

template<
    int in_h, int in_w, 
    int ker_h, int ker_w, 
    int c_in, int c_out, 
    int stride, 
    int max_pool_ker_h, int max_pool_ker_w, 
    int padding,
    int ch_in_unroll_factor,
    typename input_type, typename filter_type, typename potential_type>
void conv_leak_max_pool(
    input_type input[c_in][in_h][in_w],
    filter_type filter[c_out][c_in][ker_h][ker_w],
    filter_type bias[c_out],
    bit_t output[c_out][((in_h + 2 * padding - ker_h) / stride + 1) / max_pool_ker_h][((in_w + 2 * padding - ker_w) / stride + 1) / max_pool_ker_w])
{
    constexpr int in_h_with_padding = in_h + 2 * padding;
    constexpr int in_w_with_padding = in_w + 2 * padding;

    input_type input_with_padding[c_in][in_h_with_padding][in_w_with_padding];

    constexpr int out_conv_h = (in_h_with_padding - ker_h) / stride + 1;
    constexpr int out_conv_w = (in_w_with_padding - ker_w) / stride + 1;

    static potential_type potentials[c_out][out_conv_h][out_conv_w] = {};
    bit_t spikes[c_out][out_conv_h][out_conv_w] = {};

    // #pragma HLS ARRAY_PARTITION variable=potentials dim=3 type=complete

    constexpr int out_max_pool_h = out_conv_h / max_pool_ker_h;
    constexpr int out_max_pool_w = out_conv_w / max_pool_ker_w;

    add_padding<c_in, in_h, in_w, padding>(input, input_with_padding);

    conv_2d<in_h + 2 * padding, in_w + 2 * padding, ker_h, ker_w, c_in, c_out, stride, ch_in_unroll_factor>(
        input_with_padding,
        filter,
        bias,
        potentials
    );

    // print_mat<c_out>(potentials, "conv_pot");

    // ------------------------------------------------------

    bit_t temp[c_out][out_conv_h][out_conv_w];
    // #pragma HLS ARRAY_PARTITION variable=temp dim=3 type=complete

    fire_conv<c_out>(potentials, temp);

    // print_mat<c_out>(potentials, "conv_pot_after_fire");

    // print_mat<c_out>(temp, "spikes_conv");

    max_pool_spike_2d<c_out, out_conv_h, out_conv_w, 2, 2>(temp, output);

    // print_mat<c_out>(output, "max_pool_result");

            // ------------------------------------------------------

            //fire_conv_max_pool_2d<c_out, out_conv_h, out_conv_w, max_pool_ker_h, max_pool_ker_w, potential_type>(potentials, output);

            //print_mat<c_out>(potentials, "conv-pos-decay");

            //print_mat<c_out>(output, "max-pool");

            // fire_conv_2d<c_out, out_conv_h, out_conv_w, potential_t>(potentials, spikes);
            // max_pool_2d<c_out, out_conv_h, out_conv_w, max_pool_ker_h, max_pool_ker_w, bit_t>(spikes,output);
}

void scnn_tobias_fashion_mnist(input_t input[28][28], bit_t output[NUM_OUTPUTS])
{
    int const conv_ker = 3;
    int const max_pool_ker = 2;
    int const stride = 1;
    int const padding = 1;

    int const ch_in_layer_0 = 1;

    int const dim_layer_0 = 28;
    int const ch_out_layer_0 = 16;

    int const dim_layer_1 = 14;
    int const ch_out_layer_1 = 32;

    int const dim_layer_2 = 7;
    int const ch_out_layer_2 = 64;

    int const dim_layer_3 = 3;

    int const dim_flat = 64 * 3 * 3;
    int const neurons_flat_1 = 128;

    bit_t output_layer_0[ch_out_layer_0][dim_layer_1][dim_layer_1];
    bit_t output_layer_1[ch_out_layer_1][dim_layer_2][dim_layer_2];
    bit_t output_layer_2[ch_out_layer_2][dim_layer_3][dim_layer_3];

    bit_t output_layer_2_flat[dim_flat];
    bit_t output_layer_3[neurons_flat_1];

    input_t input_with_channel[1][28][28];
    add_channel_dimension(input, input_with_channel);

    conv_leak_max_pool<dim_layer_0, dim_layer_0, conv_ker, conv_ker, ch_in_layer_0, ch_out_layer_0, stride, max_pool_ker, max_pool_ker, padding, 16, input_t, weight_t, potential_t>(
        input_with_channel,
        layer_0_weight,
        layer_0_bias,
        output_layer_0
    );

    conv_leak_max_pool<dim_layer_1, dim_layer_1, conv_ker, conv_ker, ch_out_layer_0, ch_out_layer_1, stride, max_pool_ker, max_pool_ker, padding, 32, bit_t, weight_t, potential_t>(
        output_layer_0,
        layer_4_weight,
        layer_4_bias,
        output_layer_1
    );

    conv_leak_max_pool<dim_layer_2, dim_layer_2, conv_ker, conv_ker, ch_out_layer_1, ch_out_layer_2, stride, max_pool_ker, max_pool_ker, padding, 64, bit_t, weight_t, potential_t>(
        output_layer_1,
        layer_8_weight,
        layer_8_bias,
        output_layer_2
    );

    // flatten<ch_out_layer_2>(output_layer_2, output_layer_2_flat);
    // ann_lif_layer_with_bias<neurons_flat_1, dim_flat, bit_t, 1, 1>(output_layer_2_flat, output_layer_3, layer_13_weight, layer_13_bias);

    // ----------------------------------------

    // ann_lif_layer_with_bias<neurons_flat_1, dim_flat, bit_t, 32, 128>(&output_layer_2[0][0][0], output_layer_3, layer_13_weight, layer_13_bias);
    // ann_lif_layer_with_bias<NUM_OUTPUTS, neurons_flat_1, bit_t, 16, 10>(output_layer_3, output, layer_16_weight, layer_16_bias);


    ann_lif_layer_with_bias<neurons_flat_1, dim_flat, bit_t, 1, 1>(&output_layer_2[0][0][0], output_layer_3, layer_13_weight, layer_13_bias, true);

    //print_vet<128>(output_layer_3, "output-first-dense");  

    ann_lif_layer_with_bias<NUM_OUTPUTS, neurons_flat_1, bit_t, 1, 1>(output_layer_3, output, layer_16_weight, layer_16_bias, true);

    //print_vet<10>(output, "final-output");  
}

void snn_mnist_hls(input_t input[28][28], bit_t output[NUM_OUTPUTS]) 
// void snn_mnist_hls(input_ecg_t input[NUM_INPUTS], bit_t output [num_neurons[1]]) 
// void snn_mnist_hls(bit_t bit_input[NUM_INPUTS], bit_t output [num_neurons[1]]) 
// void snn_mnist_hls(hls::stream<neuron_idx_t> &input , bit_t output [num_neurons[1]])
{
    // ann_like_lascas_2026(input, output, weights_0, weights_1);

    scnn_tobias_fashion_mnist(input, output);
}