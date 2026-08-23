#include "neuro_hls_functions/bit_type.h"
#include "neuro_hls_functions/time_driven.h"
#include "quantization.h"
#include "neuron_params.h"
#include "snn_implementation.h"

void snn_to_hls(input_t (&input)[12], bit_t (&output)[7], bool reset_potentials)
{
//--------------------------------------------------
// implementation of 'fc1' layer
//--------------------------------------------------

	potential_t layer_1[38];
	AffineReuse<1>(input, layer_1, weights_1, bias_1);

//--------------------------------------------------
// implementation of 'merge_1' layer
//--------------------------------------------------

	static potential_t layer_2_rec[38] = {};
	if (reset_potentials) {
		for (int i0 = 0; i0 < 38; ++i0) {
			layer_2_rec[i0] = potential_t(0);
		}
	}
	Merge<1>(layer_1, layer_2_rec);

//--------------------------------------------------
// implementation of 'lif1.lif' layer
//--------------------------------------------------

	bit_t layer_3[38];
	CubaLIF<dynamics_t,1>(layer_1, layer_3, alpha_syn_3, beta_mem_3, r_3, v_leak_3, v_threshold_3, v_reset_3, w_in_3, u_state_3, v_state_3, reset_potentials, false);

//--------------------------------------------------
// implementation of 'lif1.w_rec' layer
//--------------------------------------------------

	AffineReuse<1>(layer_3, layer_2_rec, weights_2_rec, bias_2_rec);

//--------------------------------------------------
// implementation of 'fc2' layer
//--------------------------------------------------

	potential_t layer_4[7];
	AffineReuse<1>(layer_3, layer_4, weights_4, bias_4);

//--------------------------------------------------
// implementation of 'lif2' layer
//--------------------------------------------------

	CubaLIF<dynamics_t,1>(layer_4, output, alpha_syn_5, beta_mem_5, r_5, v_leak_5, v_threshold_5, v_reset_5, w_in_5, u_state_5, v_state_5, reset_potentials, false);
}
