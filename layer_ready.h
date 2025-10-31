#ifndef _LAYER_READY_H_
#define _LAYER_READY_H_

template<int n_neurons, int unroll_factor, typename potential_type>
void leaky_fire_dense(potential_type potentials[n_neurons], bit_t output[n_neurons])
{
    #pragma HLS INLINE
    #pragma HLS BIND_OP variable=potentials op=mul impl=dsp

    leaky_fire_dense_apply_decay:
    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=unroll_factor
        potentials[n] *= layer::decay;
    }

    leaky_fire_dense_check_threshold:
    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=unroll_factor

        if (potentials[n] >= 1)
        {
            output[n] = 1;
            potentials[n] -= layer::threshold; ///// POR ENQUANTO, SUPORTE APENAS PARA SUBTRACT EM CASO DE FIRE
        }
        else
        {
            output[n] = 0;
        }
    }
}

template<int n_neurons, int n_inputs, int unroll_factor, typename input_type, typename potential_type>
void dense(input_type input [n_inputs],
           potential_type potentials [n_neurons],
           weight_t weights [n_neurons][n_inputs],
           weight_t bias[n_neurons])
{
    potential_t aux[unroll_factor];

    #pragma HLS ARRAY_PARTITION variable=potentials dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=bias dim=1 type=complete

    #pragma HLS ARRAY_PARTITION variable=input factor=unroll_factor dim=1 type=cyclic
    #pragma HLS ARRAY_PARTITION variable=aux factor=unroll_factor dim=1 type=cyclic

    // OBS: unroll_factor precisa dividir n_inputs

    dense_inputs:
    for (int i = 0; i < n_inputs; i += unroll_factor)
    {
    #pragma HLS PIPELINE off
        
        dense_neurons:
        for (int n = 0; n < n_neurons; n++)
        {
            #pragma HLS PIPELINE off

            dense_mult_batch:
            for (int k = 0; k < unroll_factor; k++)
            {
                #pragma HLS UNROLL

                aux[k] = weights[n][i + k] * input[i + k];
                potentials[n] += aux[k];
            }
        }
    }

    dense_bias:
    for (int n = 0; n < n_neurons; n++)
    {
        #pragma HLS UNROLL factor=unroll_factor
        potentials[n] += bias[n];
    }
};

#endif