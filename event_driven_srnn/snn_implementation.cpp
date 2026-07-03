#include "bit_type.h"
#include "quantization.h"
#include "neuron_params.h"
#include "snn_implementation.h"
#include "affine.h"
#include "merge.h"
#include "cuba_lif.h"
#include "hls_stream.h"
#include "split.h"

void snn_to_hls(hls::stream<spike_t> &input, hls::stream<spike_t> &output)
{
	#pragma HLS DATAFLOW
    // Variável estática elevada ao escopo global da função para ser inicializada no início do sistema
	static hls::stream<spike_t> layer_2_rec("layer_2_rec");
    static bool initialized = false;

    // PREVENINDO DEADLOCK: Em uma rede Recorrente mapeada para hardware por stream sem delays explícitos,
    // nós forçamos um "estado inicial zerado" (engatilhado) para que o Merge aguardando no ciclo (t=0)
    // consiga ser destrancado usando o peso de (t-1).
    if (!initialized) {
        spike_t init_token;
        init_token.type = TYPE_END_STEP; 
        init_token.amplitude = 0;
        init_token.time_step = 0;
        init_token.timestamp = 0;
        layer_2_rec.write(init_token);
        initialized = true;
    }

//--------------------------------------------------
// implementation of 'fc1' layer
//--------------------------------------------------

	static hls::stream<spike_t> layer_1;
    Affine<12,38>(input, layer_1, weights_1, bias_1); // RESET BY ZERO

//--------------------------------------------------
// implementation of 'merge_1' layer
//--------------------------------------------------

	// (layer_2_rec declarada no início da função)
	static hls::stream<spike_t> layer_1_merged;
	Merge(layer_1, layer_2_rec, layer_1_merged);

//--------------------------------------------------
// implementation of 'lif1.lif' layer
//--------------------------------------------------

	static hls::stream<spike_t> layer_3;
	CubaLIF<1,1,38>(layer_1_merged, layer_3, tau_syn_3, tau_mem_3, r_3, v_leak_3, v_threshold_3, v_reset_3, w_in_3);

//--------------------------------------------------
// implementation of 'split_1' layer
//--------------------------------------------------

	static hls::stream<spike_t> layer_3_split1;
	static hls::stream<spike_t> layer_3_split2;
	Split(layer_3, layer_3_split1, layer_3_split2);

//--------------------------------------------------
// implementation of 'lif1.w_rec' layer
//--------------------------------------------------

	Affine<38,38>(layer_3_split1, layer_2_rec, weights_2_rec, bias_2_rec); // RESET BY ZERO

//--------------------------------------------------
// implementation of 'fc2' layer
//--------------------------------------------------

	static hls::stream<spike_t> layer_4;
    Affine<38,7>(layer_3_split2, layer_4, weights_4, bias_4); // RESET BY ZERO
	
//--------------------------------------------------
// implementation of 'lif2' layer
//--------------------------------------------------

	CubaLIF<1,1,7>(layer_4, output, tau_syn_5, tau_mem_5, r_5, v_leak_5, v_threshold_5, v_reset_5, w_in_5);
}
