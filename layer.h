#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <asm-generic/errno.h>
#include <iomanip>
#include <vector>
#include <ap_fixed.h>
#include <ap_int.h>
#include <string>
#include <fstream>
#include <iostream>
#include <exception>
#include <hls_stream.h>
#include <hls_vector.h>
#include <bitset>

#include "types_and_params.h"

using namespace std;

template<int n_neurons, int accum_unroll_factor>
void fire_array(potential_t potentials[n_neurons], bit_t output[n_neurons])
{
    #pragma HLS INLINE
    #pragma HLS BIND_OP variable=potentials op=mul impl=dsp

    fire_array:
    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=accum_unroll_factor
        if (potentials[n] >= 1)
        {
            output[n] = 1;
            potentials[n] = 0;
        }
        else
        {
            output[n] = 0;
            potentials[n] *= layer::decay;
            // potentials[n] >>= 1;
        }
    }
}

template<int n_neurons, int fire_unroll_factor>
void fire_array_decay_before(potential_t potentials[n_neurons], bit_t output[n_neurons])
{
    #pragma HLS INLINE
    #pragma HLS BIND_OP variable=potentials op=mul impl=dsp

    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=fire_unroll_factor
        potentials[n] *= layer::decay;
    }

    fire_array_decay_before:
    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=fire_unroll_factor

        if (potentials[n] >= 1)
        {
            output[n] = 1;
            potentials[n] -= layer::threshold;
        }
        else
        {
            output[n] = 0;
        }
    }
}

template<int n_neurons, int n_inputs, typename input_ty, int accum_unroll_factor, int fire_unroll_factor>
void ann_lif_layer(input_ty input [n_inputs],
               bit_t output [n_neurons],
               weight_t weights [n_neurons][n_inputs])
{
#pragma HLS function_instantiate variable=input

    static potential_t potentials[n_neurons] = {};

    potential_t aux[accum_unroll_factor];

    // #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=output dim=1 type=complete

    // #pragma HLS ARRAY_PARTITION variable=input factor=accum_unroll_factor dim=1 type=cyclic
    // #pragma HLS ARRAY_PARTITION variable=aux factor=accum_unroll_factor dim=1 type=cyclic

    // OBS: accum_unroll_factor precisa dividir n_inputs

    itter_inputs:
    for (int i = 0; i < n_inputs; i += accum_unroll_factor)
    {
    #pragma HLS PIPELINE off
        
        itter_neurons:
        for (int n = 0; n < n_neurons; n++)
        {
            #pragma HLS PIPELINE off

            mult_batch:
            for (int k = 0; k < accum_unroll_factor; k++)
            {
                #pragma HLS PIPELINE off

                aux[k] = weights[n][i + k] * input[i + k];
                potentials[n] += aux[k];
            }
        }
    }

    fire_array<n_neurons, fire_unroll_factor>(potentials, output);

    // for (int i = 0; i < 10; i++)
    // {
    //     cout << potentials[i] << " ";
    // }

    // cout << endl;
};

template<int n_neurons, int n_inputs, typename input_ty, int accum_unroll_factor, int fire_unroll_factor>
void ann_lif_layer_with_bias(input_ty input [n_inputs],
                             bit_t output [n_neurons],
                             weight_t weights [n_neurons][n_inputs],
                             weight_t bias[n_neurons],
                             bool print_pot = false)
{
#pragma HLS function_instantiate variable=input

    static potential_t potentials[n_neurons] = {};

    potential_t aux[accum_unroll_factor];

    // #pragma HLS bind_storage variable=<array> type=<ram_1p|ram_2p|ram_s2p> impl=<lutram|bram|uram|ff>

    // #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=cyclic factor=accum_unroll_factor
    // #pragma HLS ARRAY_PARTITION variable=output dim=1 type=cyclic factor=accum_unroll_factor
    // #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=cyclic factor=accum_unroll_factor

    #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=output dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=complete

    #pragma HLS ARRAY_PARTITION variable=input factor=accum_unroll_factor dim=1 type=cyclic
    #pragma HLS ARRAY_PARTITION variable=aux factor=accum_unroll_factor dim=1 type=cyclic

    // #pragma HLS BIND_STORAGE variable=weights type=rom_2p impl=lutram
    // #pragma HLS BIND_STORAGE variable=weights type=ram_2p impl=uram

    // OBS: accum_unroll_factor precisa dividir n_inputs

    dense_inputs:
    for (int i = 0; i < n_inputs; i += accum_unroll_factor)
    {
    #pragma HLS PIPELINE off
        
        dense_neurons:
        for (int n = 0; n < n_neurons; n++)
        {
            // #pragma HLS UNROLL factor=2
            #pragma HLS PIPELINE off

            dense_mult_batch:
            for (int k = 0; k < accum_unroll_factor; k++)
            {
                #pragma HLS UNROLL
                // #pragma HLS PIPELINE off

                aux[k] = weights[n][i + k] * input[i + k];
                potentials[n] += aux[k];
            }
        }
    }

    dense_bias:
    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=accum_unroll_factor
        // #pragma HLS PIPELINE off
        potentials[n] += bias[n];
    }

    fire_array_decay_before<n_neurons, fire_unroll_factor>(potentials, output);
    // fire_array<n_neurons, fire_unroll_factor>(potentials, output);

    /*if (print_pot)
    {
        //print_vet<n_neurons>(potentials, "dense-pot");
    }*/
};

template<int n_neurons, int n_inputs, typename input_ty, int accum_unroll_factor, int fire_unroll_factor, int print>
void ann_lif_layer_debug(input_ty input [n_inputs],
                         bit_t output [n_neurons],
                         weight_t weights [n_neurons][n_inputs])
{
#pragma HLS function_instantiate variable=input

    static potential_t potentials[n_neurons] = {};

    potential_t aux[accum_unroll_factor];

    // #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=output dim=1 type=complete

    // #pragma HLS ARRAY_PARTITION variable=input factor=accum_unroll_factor dim=1 type=cyclic
    // #pragma HLS ARRAY_PARTITION variable=aux factor=accum_unroll_factor dim=1 type=cyclic

    // OBS: accum_unroll_factor precisa dividir n_inputs

    //cout << "PRE-FIRE: ";

    // cout << "POS-FIRE: ";
    // for (int i = 0; i < print; i++)
    // {
    //     cout << potentials[i] << " ";
    // }
    // cout << endl << endl;

    itter_inputs:
    for (int i = 0; i < n_inputs; i += accum_unroll_factor)
    {
    #pragma HLS PIPELINE off
        
        itter_neurons:
        for (int n = 0; n < n_neurons; n++)
        {
            #pragma HLS PIPELINE off

            mult_batch:
            for (int k = 0; k < accum_unroll_factor; k++)
            {
                #pragma HLS PIPELINE off

                aux[k] = weights[n][i + k] * input[i + k];
                potentials[n] += aux[k];
            }
        }
    }

    // cout << "PRE-FIRE: ";

    for (int i = 0; i < print; i++)
    {
        cout << potentials[i] << " ";
    }
    cout << endl;

    fire_array<n_neurons, fire_unroll_factor>(potentials, output);

    // cout << "POS-FIRE: ";
    // for (int i = 0; i < print; i++)
    // {
    //     cout << potentials[i] << " ";
    // }
    // cout << endl;
};

// template<int n_neurons, int n_inputs, typename input_t, int accum_unroll_factor, int fire_unroll_factor>
// void ann_lif_layer(input_t input [n_inputs],
//                bit_t output [n_neurons],
//                weight_t weights [n_neurons][n_inputs])
// {
// #pragma HLS function_instantiate variable=input

//     static potential_t potentials[n_neurons] = {};

//     potential_t aux[accum_unroll_factor];

//     #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=complete
//     #pragma HLS ARRAY_PARTITION variable=output dim=1 type=complete

//     #pragma HLS ARRAY_PARTITION variable=input factor=accum_unroll_factor dim=1 type=cyclic
//     #pragma HLS ARRAY_PARTITION variable=aux factor=accum_unroll_factor dim=1 type=cyclic

//     fire_array<n_neurons, fire_unroll_factor>(potentials, output);

//     // OBS: accum_unroll_factor precisa dividir n_inputs

//     itter_inputs:
//     for (int i = 0; i < n_inputs; i += accum_unroll_factor)
//     {
//     #pragma HLS PIPELINE II=2
        
//         itter_neurons:
//         for (int n = 0; n < n_neurons; n++)
//         {
//             #pragma HLS UNROLL

//             mult_batch:
//             for (int k = 0; k < accum_unroll_factor; k++)
//             {
//                 #pragma HLS UNROLL

//                 aux[k] = weights[n][i + k] * input[i + k];
//                 potentials[n] += aux[k];
//             }
//         }
//     }
// };

/*
    c_in: canais de entrada
    l_in: número de elementos de cada canal
    c_out: canais de saida (numero de filtros)
    
    OBS:
        1) l_out é calculado automaticamente com base no l_in, c_out e kernel_size
        2) O parâmetro de dilatation é considerado sempre como 1 (elementos consecutivos
           serão usados pelo kernel)
        3) Sem suporte para padding (por enquanto)
*/
template<int c_in, int l_in, int c_out, int kernel_size, int stride, typename input_ty>
void conv_1d(input_ty input[c_in][l_in], float filter[c_out][c_in][kernel_size], float result[c_out][(l_in - kernel_size) / stride + 1])
{
    constexpr int l_out = (l_in - kernel_size) / stride + 1;
    // cout << "l_out: " << l_out << endl;

    // input_t result[c_out][l_out] = {};
    int kernel_start;

    ch_out: for (int c = 0; c < c_out; c++)
    {
        #pragma HLS PIPELINE off
        kernel_start = 0;

        res_l_out: for (int l = 0; l < l_out; l++)
        {
            #pragma HLS PIPELINE off

            ch_in: for (int i = 0; i < c_in; i++)
            {
                #pragma HLS PIPELINE off
                kernel: for (int k = 0; k < kernel_size; k++)
                {
                    #pragma HLS UNROLL
                    //cout << filter[c][i][k] << " * " << input[i][kernel_start + k] << " = " << filter[c][i][k] * input[i][kernel_start + k] << endl;
                    result[c][l] += filter[c][i][k] * input[i][kernel_start + k];
                }
            }

            //cout << "[" << c << "][" << l << "] = " << result[c][l] << endl;

            kernel_start += stride;
        }
    }

    // cout << "Result:" << endl;

    // for (int c = 0; c < c_out; c++)
    // {
    //     for (int l = 0; l < l_out; l++)
    //     {
    //         cout << result[c][l] << " ";
    //     }

    //     cout << endl;
    // }
}

/*
    in_w: largura da entrada
    in_h: altura da entrada

    ker_w: largura do kernel
    ker_h: altura do kernel

    c_in: canais de entrada
    c_out: canais de saída

    OBS:
        1) As dimensões da saída são calculadas automaticamente
        2) O parâmetro de dilatation é considerado sempre como 1 (elementos consecutivos
           serão usados pelo kernel)
        3) Sem suporte para padding (por enquanto)
        4) Só há suporte para stride horizontal = stride vertical (por enquanto)
*/
template<int in_h, int in_w, int ker_h, int ker_w, int c_in, int c_out, int stride, int ch_out_unroll_factor, typename input_ty, typename filter_t, typename res_t>
void conv_2d(input_ty input[c_in][in_h][in_w],
             filter_t filter[c_out][c_in][ker_h][ker_w],
             filter_t bias[c_out],
             res_t result[c_out][(in_h - ker_h) / stride + 1][(in_w - ker_w) / stride + 1])
{
    constexpr int out_h = (in_h - ker_h) / stride + 1;
    constexpr int out_w = (in_w - ker_w) / stride + 1;

    res_t aux[c_out][c_in];

    // #pragma HLS ARRAY_PARTITION variable=filter dim=0 type=complete

    #pragma HLS ARRAY_PARTITION variable=filter dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=filter dim=2 type=complete

    // #pragma HLS ARRAY_PARTITION variable=input dim=0 type=complete
    #pragma HLS ARRAY_PARTITION variable=input dim=1 type=complete

    #pragma HLS ARRAY_PARTITION variable=bias dim=0 type=complete
    #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete

    #pragma HLS ARRAY_PARTITION variable=aux dim=0 type=complete 

    char ker_start_h = 0;
    char ker_start_w = 0;

    conv_lines_out:
    for (int res_l = 0; res_l < out_h; res_l++)
    {
        #pragma HLS PIPELINE off

        ker_start_w = 0;

        conv_cols_out:
        for (int res_c = 0; res_c < out_w; res_c++)
        {
            #pragma HLS PIPELINE off

            make_aux_zero:
            for (int j = 0; j < c_in; j++)
            {
                // #pragma HLS PIPELINE off
                #pragma HLS UNROLL

                for (int i = 0; i < c_out; i++)
                {
                    // #pragma HLS PIPELINE off
                    #pragma HLS UNROLL
                    aux[i][j] = 0;
                }
            }

            //// NAO ESQUECER DE COLOCAR UNROLL DE VOLTA

            conv_ker_lines:
            for (int kh = 0; kh < ker_h; kh++)
            {
                // #pragma HLS UNROLL
                #pragma HLS PIPELINE off

                conv_ker_cols:
                for (int kw = 0; kw < ker_w; kw++)
                {
                    // #pragma HLS UNROLL
                    #pragma HLS PIPELINE off        

                    conv_ch_in:
                    for (int in_ch = 0; in_ch < c_in; in_ch++)
                    {
                        #pragma HLS UNROLL
                        // #pragma HLS PIPELINE off
                        
                        conv_ch_out:
                        for (int out_ch = 0; out_ch < c_out; out_ch++)
                        {
                            #pragma HLS UNROLL
                            // #pragma HLS UNROLL factor=ch_out_unroll_factor
                            // #pragma HLS PIPELINE off
                            aux[out_ch][in_ch] += input[in_ch][ker_start_h + kh][ker_start_w + kw] * filter[out_ch][in_ch][kh][kw];
                        }
                    }
                }
            }

            add_bias_to_result:
            for (int i = 0; i < c_out; i++)
            {
                // #pragma HLS PIPELINE off
                #pragma HLS UNROLL
                result[i][res_l][res_c] += bias[i];
            }

            add_aux_result_c_in:
            for (int j = 0; j < c_in; j++)            
            {
                #pragma HLS PIPELINE off

                add_aux_result_c_out:
                for (int i = 0; i < c_out; i++)
                {
                    // #pragma HLS PIPELINE off
                    #pragma HLS UNROLL
                    result[i][res_l][res_c] += aux[i][j];
                }
            }

            ker_start_w++;
        }

        ker_start_h++;
    }
}

template<int in_h, int in_w, int ker_h, int ker_w, int c_in, int c_out, int stride, int lixo, typename input_ty, typename filter_t, typename res_t>
void conv_2d_old(input_ty input[c_in][in_h][in_w],
             filter_t filter[c_out][c_in][ker_h][ker_w],
             filter_t bias[c_out],
             res_t result[c_out][(in_h - ker_h) / stride + 1][(in_w - ker_w) / stride + 1])
{
    constexpr int out_h = (in_h - ker_h) / stride + 1;
    constexpr int out_w = (in_w - ker_w) / stride + 1;

    // #pragma HLS ARRAY_PARTITION variable=input dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=filter dim=1 type=complete

    res_t aux;
    char ker_start_h = 0;
    char ker_start_w = 0;

    conv_lines_out:
    for (int res_l = 0; res_l < out_h; res_l++)
    {
        #pragma HLS PIPELINE off
        ker_start_w = 0;

        conv_cols_out:
        for (int res_c = 0; res_c < out_w; res_c++)
        {
            #pragma HLS PIPELINE off

            conv_ch_out:
            for (int out_ch = 0; out_ch < c_out; out_ch++)
            {
                // #pragma HLS UNROLL /*Se remover, 300.000 ciclos*/
                #pragma HLS PIPELINE off
                aux = 0;

                conv_ch_in:
                for (int in_ch = 0; in_ch < c_in; in_ch++)
                {
                    // #pragma HLS UNROLL
                    #pragma HLS PIPELINE off
                    
                    conv_ker_lines:
                    for (int kh = 0; kh < ker_h; kh++)
                    {
                        // #pragma HLS UNROLL
                        #pragma HLS PIPELINE off
                        
                        conv_ker_cols:
                        for (int kw = 0; kw < ker_w; kw++)
                        {
                            // #pragma HLS UNROLL
                            #pragma HLS PIPELINE off
                            aux += input[in_ch][ker_start_h + kh][ker_start_w + kw] * filter[out_ch][in_ch][kh][kw];
                        }
                    }
                }

                result[out_ch][res_l][res_c] += aux + bias[out_ch];
            }

            ker_start_w++;
        }

        ker_start_h++;
    }
}

template<int c_in, int in_h, int in_w, typename input_ty>
void fire_conv_2d(input_ty potentials[c_in][in_h][in_w], bit_t output[c_in][in_h][in_w])
{
    fire_ch_in:
    for (int ch = 0; ch < c_in; ch++)
    {
        #pragma HLS PIPELINE off

        fire_lines:
        for (int h = 0; h < in_h; h++)
        {
            #pragma HLS PIPELINE OFF

            fire_cols:
            for (int w = 0; w < in_w; w++)
            {
                #pragma HLS UNROLL
                if (potentials[ch][h][w] > THRESHOLD)
                {
                    output[ch][h][w] = 1;
                    potentials[ch][h][w] = 0;
                }
                else
                {
                    output[ch][h][w] = 0;
                    potentials[ch][h][w] *= layer::decay;
                }
            }
        }
    }
}

// template<int c_in, int in_h, int in_w, int ker_h, int ker_w, typename data_type>
// void max_pool_2d(data_type input[c_in][in_h][in_w], data_type result[c_in][in_h / ker_h][in_w / ker_w])
// {
//     constexpr int out_h = in_h / ker_h;
//     constexpr int out_w = in_w / ker_w;

//     int ker_start_h;
//     int ker_start_w;

//     max_ch_in:
//     for (int ch = 0; ch < c_in; ch++)
//     {
//         #pragma HLS PIPELINE off
//         ker_start_h = 0;

//         max_out_lines:
//         for (int i = 0; i < out_h; i++)
//         {
//             #pragma HLS PIPELINE off
//             ker_start_w = 0;

//             max_out_cols:
//             for (int j = 0; j < out_w; j++)
//             {
//                 #pragma HLS PIPELINE off
//                 int max_value = input[ch][ker_start_h][ker_start_w];
                
//                 max_ker_lines:
//                 for (int ker_i = 0; ker_i < ker_h; ker_i++)
//                 {
//                     #pragma HLS PIPELINE off
                    
//                     max_ker_cols:
//                     for (int ker_j = 1; ker_j < ker_w; ker_j++)
//                     {
//                         #pragma HLS PIPELINE off
//                         if (input[ch][ker_start_h + ker_i][ker_start_w + ker_j] > max_value)
//                         {
//                             max_value = input[ch][ker_start_h + ker_i][ker_start_w + ker_j];
//                         }
//                     }
//                 }

//                 result[ch][i][j] = max_value;
//                 ker_start_w += ker_w;
//             }

//             ker_start_h += ker_h;
//         }
//     }
// }

template<int c_in, int in_h, int in_w, typename input_type>
void fire_conv_old(input_type potentials[c_in][in_h][in_w], bit_t output[c_in][in_h][in_w])
{
    for (int ch = 0; ch < c_in; ch++)
    {
        #pragma HLS PIPELINE off

        for (int l = 0; l < in_h; l++)
        {
            #pragma HLS PIPELINE off

            for (int c = 0; c < in_w; c++)
            {
                #pragma HLS PIPELINE off

                potentials[ch][l][c] *= layer::decay;

                if (potentials[ch][l][c] >= layer::threshold)
                {
                    output[ch][l][c] = 1;
                }
                else
                {
                    output[ch][l][c] = 0;
                }
            }
        }
    }

    //print_mat<c_in>(output, "spikes of conv");
}

template<int c_in, int in_h, int in_w, typename input_type>
void fire_conv(input_type potentials[c_in][in_h][in_w], bit_t output[c_in][in_h][in_w])
{
    #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=output dim=1 type=complete

    fire_conv_lines:
    for (int l = 0; l < in_h; l++)
    {
        #pragma HLS PIPELINE off

        fire_conv_cols:
        for (int c = 0; c < in_w; c++)
        {
            #pragma HLS PIPELINE off

            fire_conv_decay:
            for (int ch = 0; ch < c_in; ch++)
            {
                #pragma HLS UNROLL
                potentials[ch][l][c] *= layer::decay;
            }

            fire_conv_check_threshold:
            for (int ch = 0; ch < c_in; ch++)
            {
                #pragma HLS UNROLL

                if (potentials[ch][l][c] >= layer::threshold)
                {
                    output[ch][l][c] = 1;
                    potentials[ch][l][c] -= layer::threshold;
                }
                else
                {
                    output[ch][l][c] = 0;
                }
            }
        }
    }

    //print_mat<c_in>(output, "spikes of conv");
}

template<int c_in, int in_h, int in_w, int ker_h, int ker_w, typename input_type>
void fire_conv_max_pool_2d(input_type potentials[c_in][in_h][in_w], bit_t output[c_in][in_h / ker_h][in_w / ker_w])
{
    // #pragma HLS ARRAY_PARTITION variable=potentials type=complete
    // #pragma HLS ARRAY_PARTITION variable=output type=complete

    constexpr int out_h = in_h / ker_h;
    constexpr int out_w = in_w / ker_w;

    #pragma HLS ARRAY_PARTITION variable=potentials dim=0 type=complete
    #pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

    bit_t one = 1;

    bit_t pot_spk[c_in][in_h][in_w] = {};

    int ker_start_h;
    int ker_start_w;

    for (int ch = 0; ch < c_in; ch++)
    {
        for (int l = 0; l < out_h; l++)
        {
            for (int c = 0; c < out_w; c++)
            {
                output[ch][l][c] = 0;
            }
        }
    }

    max_ch_in:
    for (int ch = 0; ch < c_in; ch++)
    {
        #pragma HLS UNROLL factor=4
        ker_start_h = 0;

        max_out_lines:
        for (int i = 0; i < out_h; i++)
        {
            #pragma HLS PIPELINE off
            ker_start_w = 0;

            max_out_cols:
            for (int j = 0; j < out_w; j++)
            {
                #pragma HLS PIPELINE
                
                max_ker_lines:
                for (int ker_i = 0; ker_i < ker_h; ker_i++)
                {
                    #pragma HLS UNROLL
                    
                    max_ker_cols:
                    for (int ker_j = 1; ker_j < ker_w; ker_j++)
                    {
                        #pragma HLS UNROLL

                        potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] *= layer::decay;

                        if (potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] >= layer::threshold)
                        {
                            output[ch][i][j] |= one;
                            potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] -= layer::threshold;

                            pot_spk[ch][ker_start_h + ker_i][ker_start_w + ker_j] = one;
                            //bit_out[ch][ker_start_h + ker_i][ker_start_w + ker_j] = 1;

                            //cout << "  Spike: " << ch << ", " << ker_start_h + ker_i << ", " << ker_start_w + ker_j << " -> " << i << " " << j << endl;
                        }

                        // potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] *= layer::decay;

                        // if (potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] > layer::threshold)
                        // {
                        //     output[ch][i][j] |= one;
                        //     potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] -= layer::threshold;
                        // }
                        // else
                        // {
                        //     //output[ch][h][w] = 0;
                        //     potentials[ch][ker_start_h + ker_i][ker_start_w + ker_j] *= layer::decay;
                        // }

                        // if (input[ch][ker_start_h + ker_i][ker_start_w + ker_j] > max_value)
                        // {
                        //     max_value = input[ch][ker_start_h + ker_i][ker_start_w + ker_j];
                        // }
                    }
                }

                // result[ch][i][j] = max_value;
                ker_start_w += ker_w;
            }

            ker_start_h += ker_h;
        }
    }

    //print_mat<c_in>(pot_spk, "spikes - potentials");
}

template<int in_h, int in_w, typename input_type>
void add_channel_dimension(input_type input[in_h][in_w], input_type result[1][in_h][in_w])
{
    /// NO FUTURO, MUDAR CONV_2D PARA NAO PRECISAR FAZER ISSO
    
    add_ch_lines:
    for (int h = 0; h < in_h; h++)
    {
        #pragma HLS PIPELINE off

        add_ch_cols:
        for (int w = 0; w < in_w; w++)
        {
            #pragma HLS PIPELINE off
            result[0][h][w] = input[h][w];
        }
    }
}

template<int c_in, int in_h, int in_w, int padding, typename input_type>
void add_padding(input_type input[c_in][in_h][in_w], input_type result[c_in][in_h + 2 * padding][in_w + 2 * padding])
{
    constexpr int res_h = in_h + 2 * padding;
    constexpr int res_w = in_w + 2 * padding;

    /// POR ENQUANTO, ESTOU CONSIDERANDO APENAS PADDING = 1

    #pragma HLS ARRAY_PARTITION variable=result dim=0 type=complete
    #pragma HLS ARRAY_PARTITION variable=input dim=0 type=complete

    pad_ch_in:
    for (int ch = 0; ch < c_in; ch++)
    {
        #pragma HLS PIPELINE off

        pad_lines:
        for (int h = 0; h < res_h; h++)
        {
            #pragma HLS PIPELINE off
            // #pragma HLS UNROLL
            result[ch][h][0] = 0;
            result[ch][h][res_w - 1] = 0;
        }

        pad_cols:
        for (int w = 0; w < res_w; w++)
        {
            // #pragma HLS UNROLL
            #pragma HLS PIPELINE off
            result[ch][0][w] = 0;
            result[ch][res_h - 1][w] = 0;
        }

        pad_copy_lines:
        for (int h = 0; h < in_h; h++)
        {
            // #pragma HLS UNROLL
            #pragma HLS PIPELINE off
            
            pad_copy_cols:
            for (int w = 0; w < in_w; w++)
            {
                // #pragma HLS UNROLL
                #pragma HLS PIPELINE off
                result[ch][padding + h][padding + w] = input[ch][h][w];
            }
        }
    }
}

template<int c_in, int in_h, int in_w, typename input_type>
void flatten(input_type input[c_in][in_h][in_w], input_type result[c_in * in_h * in_w])
{
    constexpr int ch_data = in_h * in_w;

    flat_ch_in:
    for (int ch = 0; ch < c_in; ch++)
    {
        #pragma HLS PIPELINE off
        
        flat_lines:
        for (int h = 0; h < in_h; h++)
        {
            #pragma HLS PIPELINE off

            flat_cols:
            for (int w = 0; w < in_w; w++)
            {
                #pragma HLS PIPELINE off
                result[ch * ch_data + h * in_w + w] = input[ch][h][w];
            }
        }
    }
}

template<int in_h, int in_w, int ker_h, int ker_w, int c_in, int c_out, int stride, typename input_ty, typename filter_t>
void conv_2d_opt(input_ty input[c_in][in_h][in_w],
                 filter_t filter[c_out][c_in][ker_h][ker_w],
                 bit_t result[c_out][(in_h - ker_h) / stride + 1][(in_w - ker_w) / stride + 1])
{
    constexpr int out_h = (in_h - ker_h) / stride + 1;
    constexpr int out_w = (in_w - ker_w) / stride + 1;

    potential_t potentials[c_out][out_h][out_w] = {};

    int aux;
    int ker_start_h = 0;
    int ker_start_w = 0;

    for (int res_l = 0; res_l < out_h; res_l++)
    {
        ker_start_w = 0;

        for (int res_c = 0; res_c < out_w; res_c++)
        {
            for (int out_ch = 0; out_ch < c_out; out_ch++)
            {
                aux = 0;

                for (int in_ch = 0; in_ch < c_in; in_ch++)
                {
                    for (int kh = 0; kh < ker_h; kh++)
                    {
                        for (int kw = 0; kw < ker_w; kw++)
                        {
                            aux += input[in_ch][ker_start_h + kh][ker_start_w + kw] * filter[out_ch][in_ch][kh][kw];
                        }
                    }
                }

                potentials[out_ch][res_l][res_c] += aux;
            }

            ker_start_w++;
        }

        ker_start_h++;
    }

    for (int out_ch = 0; out_ch < c_out; out_ch++)
    {
        for (int h = 0; h < out_h; h++)
        {
            for (int w = 0; w < out_w; w++)
            {
                if (potentials[out_ch][h][w] > THRESHOLD)
                {
                    result[out_ch][h][w] = 1;
                }
                else
                {
                    result[out_ch][h][w] = 0;
                    potentials[out_ch][h][w] *= layer::decay;
                }
            }
        }
    }
}

template<int c_in, int in_h, int in_w, int ker_h, int ker_w>
void max_pool_spike_2d_old(bit_t input[c_in][in_h][in_w], bit_t result[c_in][in_h / ker_h][in_w / ker_w])
{
    constexpr int out_h = in_h / ker_h;
    constexpr int out_w = in_w / ker_w;

    int ker_start_h;
    int ker_start_w;

    max_ch_in:
    for (int ch = 0; ch < c_in; ch++)
    {
        #pragma HLS PIPELINE off

        for (int start_line = 0; start_line < out_h; start_line++)
        {
            #pragma HLS PIPELINE off

            for (int start_col = 0; start_col < out_w; start_col++)
            {
                #pragma HLS PIPELINE off

                bit_t max_value = 0;

                for (int kh = 0; kh < ker_h; kh++)
                {
                    #pragma HLS PIPELINE off

                    for (int kw = 0; kw < ker_w; kw++)
                    {
                        #pragma HLS PIPELINE off

                        if (input[ch][start_line * ker_h + kh][start_col * ker_w + kw] == 1)
                        {
                            max_value = 1;
                        }
                    }
                }

                result[ch][start_line][start_col] = max_value;
            }
        }        
    }
}

template<int c_in, int in_h, int in_w, int ker_h, int ker_w>
void max_pool_spike_2d(bit_t input[c_in][in_h][in_w], bit_t result[c_in][in_h / ker_h][in_w / ker_w])
{
    constexpr int out_h = in_h / ker_h;
    constexpr int out_w = in_w / ker_w;

    int ker_start_h;
    int ker_start_w;

    bit_t max_value[c_in];

    #pragma HLS ARRAY_PARTITION variable=max_value dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=input dim=1 type=complete

    max_pool_lines:
    for (int start_line = 0; start_line < out_h; start_line++)
    {
        #pragma HLS PIPELINE off

        max_pool_cols:
        for (int start_col = 0; start_col < out_w; start_col++)
        {
            #pragma HLS PIPELINE off

            for (int ch = 0; ch < c_in; ch++)
            {
                // #pragma HLS PIPELINE off
                #pragma HLS UNROLL
                max_value[ch] = 0;
            }

            //// NAO ESQUECER UNROLL EM TODOS
            max_pool_ker_lines:
            for (int kh = 0; kh < ker_h; kh++)
            {
                #pragma HLS PIPELINE off
                // #pragma HLS UNROLL

                max_pool_ker_cols:
                for (int kw = 0; kw < ker_w; kw++)
                {
                    #pragma HLS PIPELINE off
                    // #pragma HLS UNROLL

                    max_ch_in:
                    for (int ch = 0; ch < c_in; ch++)
                    {
                        // #pragma HLS PIPELINE off
                        #pragma HLS UNROLL
                        max_value[ch] |= input[ch][start_line * ker_h + kh][start_col * ker_w + kw];
                    }
                }
            }

            for (int ch = 0; ch < c_in; ch++)
            {
                #pragma HLS UNROLL
                result[ch][start_line][start_col] = max_value[ch];
            }
        }
    }
}


// template<int c_in, int in_h, int in_w, int ker_h, int ker_w, typename data_type>
// void max_pool_2d(data_type input[c_in][in_h][in_w], data_type result[c_in][in_h / ker_h][in_w / ker_w])
// {
//     constexpr int out_h = in_h / ker_h;
//     constexpr int out_w = in_w / ker_w;

//     int ker_start_h;
//     int ker_start_w;

//     max_ch_in:
//     for (int ch = 0; ch < c_in; ch++)
//     {
//         #pragma HLS PIPELINE off
//         ker_start_h = 0;

//         max_out_lines:
//         for (int i = 0; i < out_h; i++)
//         {
//             #pragma HLS PIPELINE off
//             ker_start_w = 0;

//             max_out_cols:
//             for (int j = 0; j < out_w; j++)
//             {
//                 #pragma HLS PIPELINE off
//                 int max_value = input[ch][ker_start_h][ker_start_w];
                
//                 max_ker_lines:
//                 for (int ker_i = 0; ker_i < ker_h; ker_i++)
//                 {
//                     #pragma HLS PIPELINE off
                    
//                     max_ker_cols:
//                     for (int ker_j = 1; ker_j < ker_w; ker_j++)
//                     {
//                         #pragma HLS PIPELINE off
//                         if (input[ch][ker_start_h + ker_i][ker_start_w + ker_j] > max_value)
//                         {
//                             max_value = input[ch][ker_start_h + ker_i][ker_start_w + ker_j];
//                         }
//                     }
//                 }

//                 result[ch][i][j] = max_value;
//                 ker_start_w += ker_w;
//             }

//             ker_start_h += ker_h;
//         }
//     }
// }

template<int kernel, int input_height, int input_width, int n_channels>
void max_pool_2d_spike(bit_t input[n_channels][input_height][input_width], bit_t output[n_channels][input_height / kernel][input_width / kernel])
{
    constexpr int out_width = input_width / kernel;
    constexpr int out_height = input_height / kernel;

    int w_offset = 0;
    int h_offset = 0;

    for (int h = 0; h < out_height; h++)
    {
        w_offset = 0;

        for (int w = 0; w < out_width; w++)
        {
            for (int ker_h = 0; ker_h < kernel; ker_h++)
            {
                for (int ker_w = 0; ker_w < kernel; ker_w++)
                {
                    for (int ch = 0; ch < n_channels; ch++) // UNROLL COMPLETO
                    {
                        output[ch][h][w] |= input[ch][h_offset + ker_h][w_offset + ker_w]; // corrigir indices do input 
                    }
                }
            }

            w_offset += kernel;
        }

        h_offset += kernel;
    }
}

void snn_mnist_hls(input_t input[28][28], bit_t output[NUM_OUTPUTS]); 
// void snn_mnist_hls(input_ecg_t input[NUM_INPUTS], bit_t output [num_neurons[1]]);
// void snn_mnist_hls(bit_t input[NUM_INPUTS], bit_t output [num_neurons[1]]);
// void snn_mnist_hls(hls::stream<neuron_idx_t> &input, bit_t output [num_neurons[1]]);

#endif // LAYER_HPP