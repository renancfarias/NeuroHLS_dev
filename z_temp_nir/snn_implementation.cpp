#include "neuro_hls_functions/bit_type.h"
#include "quantization.h"
#include "neuron_params.h"
#include "snn_implementation.h"

void snn_to_hls(input_t input[12], bit_t output[7])
{
//--------------------------------------------------
// implementation of 'fc1' layer
//--------------------------------------------------

	potential_t layer_1[38] = {};
	Affine(input, layer_1);

//--------------------------------------------------
// implementation of 'merge_1' layer
//--------------------------------------------------

	type_t layer_2_rec[38] = {};
	Merge(layer_1, layer_2_rec);

//--------------------------------------------------
// implementation of 'lif1.lif' layer
//--------------------------------------------------

	bit_t layer_3[38] = {};
	CubaLIF(layer_1, layer_3, tau_syn_3, tau_mem_3, r_3, v_leak_3, v_threshold_3, v_reset_3, w_in_3);

//--------------------------------------------------
// implementation of 'lif1.w_rec' layer
//--------------------------------------------------

	Affine(layer_3, layer_2_rec);

//--------------------------------------------------
// implementation of 'fc2' layer
//--------------------------------------------------

	potential_t layer_4[7] = {};
	Affine(layer_3, layer_4);

//--------------------------------------------------
// implementation of 'lif2' layer
//--------------------------------------------------

	CubaLIF(layer_4, output, tau_syn_5, tau_mem_5, r_5, v_leak_5, v_threshold_5, v_reset_5, w_in_5);
}
