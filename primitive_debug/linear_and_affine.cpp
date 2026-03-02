#include <iostream>

using namespace std;

template<
    int unroll_factor,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void multiply_and_accumulate(input_type (&input)[n_inputs],
                             result_type (&result)[n_neurons],
                             params_type (&weights)[n_neurons][n_inputs])
{
    result_type aux[unroll_factor];

    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    // #pragma HLS ARRAY_PARTITION variable=aux factor=unroll_factor dim=1 type=cyclic

    dense_inputs:
    for (int i = 0; i < n_inputs; i += unroll_factor)
    {
    // #pragma HLS PIPELINE off
        
        dense_neurons:
        for (int n = 0; n < n_neurons; n++)
        {
            // #pragma HLS PIPELINE off

            dense_mult_batch:
            for (int k = 0; k < unroll_factor; k++)
            {
                // #pragma HLS UNROLL

                aux[k] = weights[n][i + k] * input[i + k];
                result[n] += aux[k];
            }
        }
    }
};

template<
    int unroll_factor,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void Linear(input_type (&input)[n_inputs],
           result_type (&result)[n_neurons],
           params_type (&weights)[n_neurons][n_inputs])
{
    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    
    dense_bias:
    for (int n = 0; n < n_neurons; n++)
    {
        // #pragma HLS UNROLL factor=unroll_factor
        result[n] = 0;
    }

    multiply_and_accumulate<unroll_factor>(input, result, weights);
};

template<
    int unroll_factor,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void Affine(input_type (&input)[n_inputs],
           result_type (&result)[n_neurons],
           params_type (&weights)[n_neurons][n_inputs],
           params_type (&bias)[n_neurons])
{
    // #pragma HLS ARRAY_PARTITION variable=result dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=complete
    // #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    
    dense_bias:
    for (int n = 0; n < n_neurons; n++)
    {
        // #pragma HLS UNROLL factor=unroll_factor
        result[n] = bias[n];
    }

    multiply_and_accumulate<unroll_factor>(input, result, weights);
};

float input_1[5] = {1, 2, 3, 4, 5};
float input_2[5] = {3, 5, 7, 9, 11};
float weights[2][5] = {{2.3, 3.2, -1.7, 2.0, 4.5}, {-0.9, 1.3, 2.6, 5.4, 0.4}};
float bias[2] = {0.5, -0.5};

int main()
{
    float out[2];
    
    cout << "Affine" << endl;
    
    Affine<1>(input_1, out, weights, bias);
    cout << "out: " << out[0] << " " << out[1] << endl;
    
    Affine<1>(input_2, out, weights, bias);
    cout << "out: " << out[0] << " " << out[1] << endl;
    
    cout << "Linear" << endl;
    
    Linear<1>(input_1, out, weights);
    cout << "out: " << out[0] << " " << out[1] << endl;
    
    Linear<1>(input_2, out, weights);
    cout << "out: " << out[0] << " " << out[1] << endl;

    return 0;
}