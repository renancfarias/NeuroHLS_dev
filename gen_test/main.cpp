void snn_to_hls(input_t input[784], bit_t output[10])
{

//--------------------------------------------------
//	Layer 1
//--------------------------------------------------

	potential_t potentials_1[128];
	bit_t spikes_1[128];

	dense<784, 128>(input, potentials_1);
	dense_LIF<128>(potentials_1, spikes_1);

//--------------------------------------------------
//	Layer 2
//--------------------------------------------------

	potential_t potentials_2[10];
	
	dense<128, 10>(spikes_1, potentials_2);
	dense_LIF<10>(potentials_2, output);
}
