#include "neuro_hls_functions/bit_type.h"
#include "neuro_hls_functions/dense.h"
#include "quantization.h"
#include "neuron_params.h"
#include "snn_implementation.h"

void snn_to_hls(input_t (&input)[2][34][34], bit_t (&output)[10], bool reset_potentials)
{
//--------------------------------------------------
// implementation of '0' layer
//--------------------------------------------------

	potential_t layer_1[16][16][16];
	Conv2d<5,5,2,2,1,1,1,1,1>(input, layer_1, weights_1, bias_1);

//--------------------------------------------------
// implementation of '1' layer
//--------------------------------------------------

	bit_t layer_2[16][16][16];
	static potential_t mem_potentials_2[16][16][16] = {};
	IF(layer_1, layer_2, mem_potentials_2, r_2, v_threshold_2, v_reset_2, reset_potentials);

//--------------------------------------------------
// implementation of '2' layer
//--------------------------------------------------

	potential_t layer_3[16][16][16];
	Conv2d<3,3,1,1,1,1,1,1,1>(layer_2, layer_3, weights_3, bias_3);

//--------------------------------------------------
// implementation of '3' layer
//--------------------------------------------------

	bit_t layer_4[16][16][16];
	static potential_t mem_potentials_4[16][16][16] = {};
	IF(layer_3, layer_4, mem_potentials_4, r_4, v_threshold_4, v_reset_4, reset_potentials);

//--------------------------------------------------
// implementation of '4' layer
//--------------------------------------------------

	potential_t layer_5[16][8][8];
	SumPool2d<2,2,2,2,0,0>(layer_4, layer_5);

//--------------------------------------------------
// implementation of '5' layer
//--------------------------------------------------

	potential_t layer_6[8][8][8];
	Conv2d<3,3,1,1,1,1,1,1,1>(layer_5, layer_6, weights_6, bias_6);

//--------------------------------------------------
// implementation of '6' layer
//--------------------------------------------------

	bit_t layer_7[8][8][8];
	static potential_t mem_potentials_7[8][8][8] = {};
	IF(layer_6, layer_7, mem_potentials_7, r_7, v_threshold_7, v_reset_7, reset_potentials);

//--------------------------------------------------
// implementation of '7' layer
//--------------------------------------------------

	potential_t layer_8[8][4][4];
	SumPool2d<2,2,2,2,0,0>(layer_7, layer_8);

//--------------------------------------------------
// implementation of '8' layer
//--------------------------------------------------

	potential_t layer_9[128];
	Flatten(layer_8, layer_9);

//--------------------------------------------------
// implementation of '9' layer
//--------------------------------------------------

	potential_t layer_10[256];
	Affine<1>(layer_9, layer_10, weights_10, bias_10);

//--------------------------------------------------
// implementation of '10' layer
//--------------------------------------------------

	bit_t layer_11[256];
	static potential_t mem_potentials_11[256] = {};
	IF(layer_10, layer_11, mem_potentials_11, r_11, v_threshold_11, v_reset_11, reset_potentials);

//--------------------------------------------------
// implementation of '11' layer
//--------------------------------------------------

	potential_t layer_12[10];
	Affine<1>(layer_11, layer_12, weights_12, bias_12);

//--------------------------------------------------
// implementation of '12' layer
//--------------------------------------------------

	static potential_t mem_potentials_13[10] = {};
	IF(layer_12, output, mem_potentials_13, r_13, v_threshold_13, v_reset_13, reset_potentials);
}
