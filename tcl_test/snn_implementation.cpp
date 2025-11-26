
#include "types_and_params.h"

#include "neuro_hls_functions/bit_type.h"
#include "neuro_hls_functions/dense.h"
#include "weights.h"

weight_t bias_1[128] = {};
weight_t bias_2[10] = {};

void snn_to_hls(input_t input[784], bit_t output[10])
{

//--------------------------------------------------
//	Layer 1
//--------------------------------------------------

	static potential_t potentials_1[128] = {};
	bit_t spikes_1[128];

	dense<784, 128, 1>(input, potentials_1, weights_0, bias_1);
	dense_LIF<128, 1>(potentials_1, spikes_1);

//--------------------------------------------------
//	Layer 2
//--------------------------------------------------

	static potential_t potentials_2[10] = {};
	
	dense<128, 10, 1>(spikes_1, potentials_2, weights_1, bias_2);
	dense_LIF<10, 1>(potentials_2, output);
}
