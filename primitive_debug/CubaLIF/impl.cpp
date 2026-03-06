#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

// Tipo bit para spikes
typedef unsigned char bit_t;

/**
 * CubaLIF Kernel - Implementação EXATA do dense.h
 */
template <typename input_type, typename W_DATA>
void cuba_lif_kernel(input_type input, W_DATA tau_syn, W_DATA w_in, W_DATA tau_mem,
                     W_DATA R, W_DATA v_leak, W_DATA dt, W_DATA v_threshold, W_DATA v_reset,
                     W_DATA& u_state, W_DATA& v_state, bit_t& spike) {
    
    // ESTÁGIO 1: Dinâmica da Sinapse (u)
    input_type leak_u = 0 - u_state;
    input_type input_u = w_in * input;
    input_type du = (dt / tau_syn) * (leak_u + input_u);
    u_state = u_state + du;
    
    // ESTÁGIO 2: Dinâmica da Membrana (v)
    input_type leak_v = v_leak - v_state;
    input_type input_v = R * u_state;
    input_type dv = (dt / tau_mem) * (leak_v + input_v);
    v_state = v_state + dv;
    
    // ESTÁGIO 3: Disparo e Reset (apenas na membrana)
    if (v_state >= v_threshold) {
        spike = 1;
        v_state = v_reset;
        // Nota: u_state NÃO é resetado em modelos CuBa
    } else {
        spike = 0;
    }
}

/**
 * CubaLIF 1D (Dense/FC Layers) - Wrapper
 */
template <typename input_type, typename W_DATA, int N>
void CubaLIF(
    const input_type (&input)[N],
    bit_t spikes_out[N],

    const W_DATA tau_syn[N],
    const W_DATA tau_mem[N],
    const W_DATA R_mem[N],
    const W_DATA v_leak[N],
    const W_DATA v_threshold[N],
    const W_DATA v_reset[N],
    const W_DATA w_in[N],

    W_DATA u_state[N],
    W_DATA v_state[N],
    const W_DATA dt
) {
    for(int i = 0; i < N; i++) {
        cuba_lif_kernel(input[i], tau_syn[i], w_in[i], tau_mem[i], R_mem[i],
                        v_leak[i], dt, v_threshold[i], v_reset[i],
                        u_state[i], v_state[i], spikes_out[i]);
    }
}

// ============================================================================
// CONFIGURAÇÃO DO TESTE
// ============================================================================

const int N_NEURONS = 3;

// Parâmetros dos neurônios
float tau_syn[N_NEURONS] = {5.0, 5.0, 5.0};
float tau_mem[N_NEURONS] = {10.0, 10.0, 10.0};
float R_mem[N_NEURONS] = {1.0, 1.0, 1.0};
float v_leak[N_NEURONS] = {0.0, 0.0, 0.0};
float v_threshold[N_NEURONS] = {1.0, 1.0, 1.0};
float v_reset[N_NEURONS] = {0.0, 0.0, 0.0};
float w_in[N_NEURONS] = {0.5, 1.0, 1.5};
float dt = 1.0;

// Estados (memória)
float u_state[N_NEURONS] = {0.0, 0.0, 0.0};
float v_state[N_NEURONS] = {0.0, 0.0, 0.0};

// Saída de spikes
bit_t spikes[N_NEURONS];

// Sequência de inputs
float input_sequence[10][N_NEURONS] = {
    {2.0, 2.0, 2.0},  // t=0
    {2.0, 2.0, 2.0},  // t=1
    {2.0, 2.0, 2.0},  // t=2
    {2.0, 2.0, 2.0},  // t=3
    {2.0, 2.0, 2.0},  // t=4
    {0.0, 0.0, 0.0},  // t=5
    {0.0, 0.0, 0.0},  // t=6
    {3.0, 3.0, 3.0},  // t=7
    {3.0, 3.0, 3.0},  // t=8
    {0.0, 0.0, 0.0},  // t=9
};

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================

void print_state(int t, float input[], ofstream& file) {
    cout << "\nTimestep " << t << ":" << endl;
    cout << "  Input: [" << input[0] << ", " << input[1] << ", " << input[2] << "]" << endl;
    cout << "  u_state: [" << fixed << setprecision(6) 
         << u_state[0] << ", " << u_state[1] << ", " << u_state[2] << "]" << endl;
    cout << "  v_state: [" << fixed << setprecision(6)
         << v_state[0] << ", " << v_state[1] << ", " << v_state[2] << "]" << endl;
    cout << "  Spikes: [" << (int)spikes[0] << ", " << (int)spikes[1] << ", " 
         << (int)spikes[2] << "]" << endl;
    
    // Salvar no arquivo
    file << "Timestep " << t << ":" << endl;
    file << "  Input: [" << fixed << setprecision(6)
         << input[0] << ", " << input[1] << ", " << input[2] << "]" << endl;
    file << "  u_state: [" << fixed << setprecision(6)
         << u_state[0] << ", " << u_state[1] << ", " << u_state[2] << "]" << endl;
    file << "  v_state: [" << fixed << setprecision(6)
         << v_state[0] << ", " << v_state[1] << ", " << v_state[2] << "]" << endl;
    file << "  Spikes: [" << (int)spikes[0] << ", " << (int)spikes[1] << ", " 
         << (int)spikes[2] << "]" << endl;
    file << endl;
}

void reset_states() {
    for (int i = 0; i < N_NEURONS; i++) {
        u_state[i] = 0.0;
        v_state[i] = 0.0;
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main()
{
    ofstream file("primitive_debug/CubaLIF/out_cpp.txt");
    
    cout << "======================================================================" << endl;
    cout << "Teste CubaLIF - Implementação C++" << endl;
    cout << "======================================================================" << endl;
    
    cout << "\nConfiguração:" << endl;
    cout << "  Neurônios: " << N_NEURONS << endl;
    cout << "  tau_syn: [" << tau_syn[0] << ", " << tau_syn[1] << ", " << tau_syn[2] << "]" << endl;
    cout << "  tau_mem: [" << tau_mem[0] << ", " << tau_mem[1] << ", " << tau_mem[2] << "]" << endl;
    cout << "  R: [" << R_mem[0] << ", " << R_mem[1] << ", " << R_mem[2] << "]" << endl;
    cout << "  v_leak: [" << v_leak[0] << ", " << v_leak[1] << ", " << v_leak[2] << "]" << endl;
    cout << "  v_threshold: [" << v_threshold[0] << ", " << v_threshold[1] << ", " << v_threshold[2] << "]" << endl;
    cout << "  v_reset: [" << v_reset[0] << ", " << v_reset[1] << ", " << v_reset[2] << "]" << endl;
    cout << "  w_in: [" << w_in[0] << ", " << w_in[1] << ", " << w_in[2] << "]" << endl;
    cout << "  dt: " << dt << endl;
    
    cout << "\n======================================================================" << endl;
    cout << "Simulação Temporal (10 timesteps)" << endl;
    cout << "======================================================================" << endl;
    
    file << "CubaLIF Test Results" << endl;
    file << "======================================================================" << endl << endl;
    
    // Executar simulação
    for (int t = 0; t < 10; t++) {
        CubaLIF<float, float, N_NEURONS>(
            input_sequence[t], spikes,
            tau_syn, tau_mem, R_mem, v_leak, v_threshold, v_reset, w_in,
            u_state, v_state, dt
        );
        
        print_state(t, input_sequence[t], file);
    }
    
    // Teste 2: Verificação de Reset
    cout << "\n======================================================================" << endl;
    cout << "Teste 2: Verificação de Reset (v_state reseta, u_state não)" << endl;
    cout << "======================================================================" << endl;
    
    reset_states();
    
    float test_input[N_NEURONS] = {5.0, 5.0, 5.0};
    
    cout << "\nInput forte: [" << test_input[0] << ", " << test_input[1] << ", " << test_input[2] << "]" << endl;
    
    for (int t = 0; t < 3; t++) {
        CubaLIF<float, float, N_NEURONS>(
            test_input, spikes,
            tau_syn, tau_mem, R_mem, v_leak, v_threshold, v_reset, w_in,
            u_state, v_state, dt
        );
        
        cout << "  t=" << t << ": u=[" << fixed << setprecision(4)
             << u_state[0] << ", " << u_state[1] << ", " << u_state[2] << "], v=["
             << v_state[0] << ", " << v_state[1] << ", " << v_state[2] << "], spike=["
             << (int)spikes[0] << ", " << (int)spikes[1] << ", " << (int)spikes[2] << "]" << endl;
    }
    
    cout << "\nObservação: Após spike, v_state vai para v_reset, mas u_state continua acumulando!" << endl;
    
    cout << "\n======================================================================" << endl;
    cout << "Resultados salvos em: primitive_debug/CubaLIF/out_cpp.txt" << endl;
    cout << "======================================================================" << endl;
    
    file.close();
    
    return 0;
}
