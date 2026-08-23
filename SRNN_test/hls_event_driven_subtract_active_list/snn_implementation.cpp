#include "neuro_hls_functions/event_driven.h"
#include "quantization.h"
#include "neuron_params.h"
#include "snn_implementation.h"

static void snn_to_hls_dataflow(hls::stream<ed_spike_t>& input_stream, hls::stream<ed_spike_t>& output_stream, hls::stream<ed_spike_t>& feedback_state_0, hls::stream<ed_spike_t>& feedback_next_0, bool reset_potentials)
{
	#pragma HLS DATAFLOW
	hls::stream<ed_spike_t> edge_1_fc2_to_lif2;
	#pragma HLS STREAM variable=edge_1_fc2_to_lif2 depth=8
	hls::stream<ed_spike_t> edge_2_lif1_lif_to_lif1_w_rec;
	#pragma HLS STREAM variable=edge_2_lif1_lif_to_lif1_w_rec depth=41
	hls::stream<ed_spike_t> edge_5_fc1_to_lif1_lif;
	#pragma HLS STREAM variable=edge_5_fc1_to_lif1_lif depth=41
	hls::stream<ed_spike_t> edge_6_lif1_lif_to_fc2;
	#pragma HLS STREAM variable=edge_6_lif1_lif_to_fc2 depth=41
	hls::stream<ed_spike_t> merge_lif1_lif_0;
	#pragma HLS STREAM variable=merge_lif1_lif_0 depth=41
	hls::stream<ed_spike_t> node_lif1_lif_out;
	#pragma HLS STREAM variable=node_lif1_lif_out depth=41
	LinearSparse<12,40,480>(input_stream, edge_5_fc1_to_lif1_lif, sparse_col_ptr_1, sparse_row_idx_1, sparse_values_1);
	Merge(edge_5_fc1_to_lif1_lif, feedback_state_0, merge_lif1_lif_0);
	CubaLIFActiveList<1,1,40,2,true>(merge_lif1_lif_0, node_lif1_lif_out, active_u_shifts_2, active_u_terms_2, active_v_shifts_2, active_v_terms_2, v_leak_2, v_threshold_2, v_reset_2, active_input_gain_2, 9.9999999999999995e-07, reset_potentials);
	Split(node_lif1_lif_out, edge_2_lif1_lif_to_lif1_w_rec, edge_6_lif1_lif_to_fc2);
	LinearSparseStep<40,40,1600>(edge_2_lif1_lif_to_lif1_w_rec, feedback_next_0, sparse_col_ptr_3, sparse_row_idx_3, sparse_values_3);
	LinearSparse<40,7,280>(edge_6_lif1_lif_to_fc2, edge_1_fc2_to_lif2, sparse_col_ptr_4, sparse_row_idx_4, sparse_values_4);
	CubaLIFActiveList<1,1,7,5,true>(edge_1_fc2_to_lif2, output_stream, active_u_shifts_5, active_u_terms_5, active_v_shifts_5, active_v_terms_5, v_leak_5, v_threshold_5, v_reset_5, active_input_gain_5, 9.9999999999999995e-07, reset_potentials);
}

void snn_to_hls(hls::stream<ed_spike_t>& input_stream, hls::stream<ed_spike_t>& output_stream, bool reset_potentials)
{
	static ed_spike_t feedback_events_0[41];
	static unsigned int feedback_size_0 = 0;
	hls::stream<ed_spike_t> feedback_state_0;
	#pragma HLS STREAM variable=feedback_state_0 depth=41
	hls::stream<ed_spike_t> feedback_next_0;
	#pragma HLS STREAM variable=feedback_next_0 depth=41
	if (reset_potentials) feedback_size_0 = 0;
	if (feedback_size_0 == 0) {
		ed_spike_t seed = {};
		seed.type = ED_TYPE_END_STEP;
		feedback_state_0.write(seed);
	} else {
		for (unsigned int i = 0; i < feedback_size_0; i++) feedback_state_0.write(feedback_events_0[i]);
	}
	snn_to_hls_dataflow(input_stream, output_stream, feedback_state_0, feedback_next_0, reset_potentials);
	bool feedback_done_0 = false;
	bool feedback_end_sample_0 = false;
	unsigned int feedback_next_size_0 = 0;
	while (!feedback_done_0) {
		ed_spike_t feedback_event_0 = feedback_next_0.read();
		if (feedback_event_0.type != ED_TYPE_SPIKE) {
			feedback_event_0.timestamp += (ed_time_step_t)NEURO_HLS_EVENT_DT;
			feedback_event_0.time_step = (unsigned int)feedback_event_0.time_step + 1;
		}
		if (feedback_next_size_0 < 41) feedback_events_0[feedback_next_size_0++] = feedback_event_0;
		feedback_end_sample_0 = feedback_event_0.type == ED_TYPE_END_SAMPLE;
		feedback_done_0 = feedback_event_0.type == ED_TYPE_END_STEP || feedback_event_0.type == ED_TYPE_END_SAMPLE;
	}
	feedback_size_0 = feedback_end_sample_0 ? 0 : feedback_next_size_0;
}
