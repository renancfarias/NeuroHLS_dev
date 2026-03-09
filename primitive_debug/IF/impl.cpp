#include <iostream>
#include <fstream>

using namespace std;

typedef bool bit_t;

/**
 * Integrate-and-Fire Kernel (copiado do dense.h para referência)
 */
template <typename input_type, typename T_PARAM>
void if_kernel(input_type input, T_PARAM R, T_PARAM threshold, input_type v_reset, 
               input_type& v_state, bool& spike) {
    #pragma HLS INLINE
    
    // 1. Integração
    v_state = v_state + (input * R);
    
    // 2. Disparo
    if (v_state >= threshold) {
        spike = true;
        v_state = v_reset;
    } else {
        spike = false;
    }
}

template <int IN_CHANNELS, int IN_H, int IN_W, typename input_type, typename params_type>
void IF(
    const input_type (&input)[IN_CHANNELS][IN_H][IN_W],
    bit_t output[IN_CHANNELS][IN_H][IN_W],

    const params_type R[IN_CHANNELS][IN_H][IN_W],
    const params_type threshold[IN_CHANNELS][IN_H][IN_W],
    const params_type v_reset[IN_CHANNELS][IN_H][IN_W])
{
    static input_type membrane_potential[IN_CHANNELS][IN_H][IN_W];

    for (int ch = 0; ch < IN_CHANNELS; ch++)
    {
        for (int h = 0; h < IN_H; h++)
        {
            for (int w = 0; w < IN_W; w++)
            {
                membrane_potential[ch][h][w] += input[ch][h][w] * R[ch][h][w];
    
                if (membrane_potential[ch][h][w] >= threshold[ch][h][w])
                {
                    output[ch][h][w] = 1;
                    membrane_potential[ch][h][w] = v_reset[ch][h][w];
                }
                else
                {
                    output[ch][h][w] = 0;
                }
            }
        }
    }
}

/**
 * Integrate-and-Fire 1D - teste da função do dense.h
 */
template <typename input_type, typename T_PARAM, int N>
void integrate_and_fire(
    const input_type input[N],
    const T_PARAM R[N],
    const T_PARAM threshold[N],
    const input_type v_reset,
    input_type membrane_potential[N],
    bool output_spikes[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        if_kernel(input[i], R[i], threshold[i], v_reset, 
                  membrane_potential[i], output_spikes[i]);
    }
}

int main()
{
    ofstream file("primitive_debug/IF/out_cpp.txt");

    // Parâmetros hardcoded - mesmos do Python
    const int N = 4;
    float R[N] = {1.0, 1.0, 1.0, 1.0};
    float threshold[N] = {4.0, 4.0, 4.0, 4.0};  // uniforme para NIR
    float v_reset = 0.0;
    
    // Estado da membrana (persiste entre timesteps)
    float membrane_potential[N] = {0.0, 0.0};
    bool spikes[N];
    
    // Inputs de teste - mesmos do Python
    float input_1[N] = {1.5, 2.0, 1.0, 2.5};
    float input_2[N] = {2.0, 2.5, 3.0, 1.5};
    float input_3[N] = {1.0, 1.5, 2.0, 1.0};
    
    file << "*** IF Neuron C++ Implementation ***" << endl;
    file << "Parameters:" << endl;
    file << "  R: [" << R[0] << ", " << R[1] << ", " << R[2] << ", " << R[3] << "]" << endl;
    file << "  Threshold: [" << threshold[0] << ", " << threshold[1] << ", " << threshold[2] << ", " << threshold[3] << "]" << endl;
    file << "  V_reset: " << v_reset << endl;
    file << endl;
    
    // Sample 1
    file << "Sample 1" << endl;
    file << "  Input: [" << input_1[0] << ", " << input_1[1] << ", " << input_1[2] << ", " << input_1[3] << "]" << endl;
    
    integrate_and_fire<float, float, N>(input_1, R, threshold, v_reset, membrane_potential, spikes);
    
    file << "  Membrane: [" << membrane_potential[0] << ", " << membrane_potential[1] << ", " << membrane_potential[2] << ", " << membrane_potential[3] << "]" << endl;
    file << "  Spikes: [" << spikes[0] << ", " << spikes[1] << ", " << spikes[2] << ", " << spikes[3] << "]" << endl;
    file << endl;
    
    // Sample 2
    file << "Sample 2" << endl;
    file << "  Input: [" << input_2[0] << ", " << input_2[1] << ", " << input_2[2] << ", " << input_2[3] << "]" << endl;
    
    integrate_and_fire<float, float, N>(input_2, R, threshold, v_reset, membrane_potential, spikes);
    
    file << "  Membrane: [" << membrane_potential[0] << ", " << membrane_potential[1] << ", " << membrane_potential[2] << ", " << membrane_potential[3] << "]" << endl;
    file << "  Spikes: [" << spikes[0] << ", " << spikes[1] << ", " << spikes[2] << ", " << spikes[3] << "]" << endl;
    file << endl;
    
    // Sample 3
    file << "Sample 3" << endl;
    file << "  Input: [" << input_3[0] << ", " << input_3[1] << ", " << input_3[2] << ", " << input_3[3] << "]" << endl;
    
    integrate_and_fire<float, float, N>(input_3, R, threshold, v_reset, membrane_potential, spikes);
    
    file << "  Membrane: [" << membrane_potential[0] << ", " << membrane_potential[1] << ", " << membrane_potential[2] << ", " << membrane_potential[3] << "]" << endl;
    file << "  Spikes: [" << spikes[0] << ", " << spikes[1] << ", " << spikes[2] << ", " << spikes[3] << "]" << endl;
    file << endl;
    
    file.close();
    
    cout << "C++ implementation test completed successfully!" << endl;
    cout << "Results saved to: primitive_debug/IF/out_cpp.txt" << endl;
    
    return 0;
}