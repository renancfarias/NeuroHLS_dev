#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int D_H, int D_W,
    int GROUPS,
    int C_IN, int H_IN, int W_IN,
    int C_OUT,
    typename input_type,
    typename result_type,
    typename params_type>
void Conv2D(
    const input_type (&input)[C_IN][H_IN][W_IN],
    result_type output[C_OUT][(H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1][(W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type weights[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type (&bias)[C_OUT])
{
    const int H_OUT = (H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1;
    const int W_OUT = (W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1;
    
    // Constantes para controle de Grupos
    const int C_IN_GROUP = C_IN / GROUPS;   // Canais de entrada por grupo
    const int C_OUT_GROUP = C_OUT / GROUPS; // Canais de saída por grupo

    // ================= LOOP PRINCIPAL =================
    
    // Itera sobre Canais de Saída (Filters)
    loop_oc: for (int oc = 0; oc < C_OUT; ++oc) {
        
        // Identifica em qual grupo este canal de saída está
        int group_id = oc / C_OUT_GROUP;
        
        // Define o intervalo de canais de entrada correspondente a este grupo
        int in_ch_start = group_id * C_IN_GROUP;

        // Itera sobre Altura da Saída
        loop_oh: for (int oh = 0; oh < H_OUT; ++oh) {
            
            // Itera sobre Largura da Saída
            loop_ow: for (int ow = 0; ow < W_OUT; ++ow) {
                #pragma HLS PIPELINE II=1
                
                // Inicializa acumulador com o BIAS
                result_type sum = (result_type)bias[oc];

                // --- Operação de Convolução ---
                
                // Itera sobre canais de entrada DO GRUPO ATUAL
                loop_ic: for (int ic_offset = 0; ic_offset < C_IN_GROUP; ++ic_offset) {
                    
                    // Canal real de entrada
                    int ic = in_ch_start + ic_offset;

                    // Itera sobre Altura do Kernel
                    loop_kh: for (int kh = 0; kh < K_H; ++kh) {
                        
                        // Itera sobre Largura do Kernel
                        loop_kw: for (int kw = 0; kw < K_W; ++kw) {
                            
                            // Cálculo da posição com DILATION e PADDING
                            // Pos = (Saida * Stride) + (Kernel * Dilation) - Padding
                            int in_row = (oh * S_H) + (kh * D_H) - P_H;
                            int in_col = (ow * S_W) + (kw * D_W) - P_W;

                            // Verificação de Borda (Padding Virtual)
                            if (in_row >= 0 && in_row < H_IN && in_col >= 0 && in_col < W_IN) {
                                sum += input[ic][in_row][in_col] * weights[oc][ic_offset][kh][kw];
                            }
                        }
                    }
                }
                
                // Escrita na saída
                output[oc][oh][ow] = sum;
            }
        }
    }
}

float weight[4][2][2][3] = {

    // ===== Grupo 1 =====

    {   // out_channel 0
        {   // in_channel 0 (do grupo 1)
            { 0.1f,  0.2f,  0.3f },
            { 0.4f,  0.5f,  0.6f }
        },
        {   // in_channel 1 (do grupo 1)
            { -0.1f, -0.2f, -0.3f },
            { -0.4f, -0.5f, -0.6f }
        }
    },

    {   // out_channel 1
        {
            { 0.7f,  0.8f,  0.9f },
            { 1.0f,  1.1f,  1.2f }
        },
        {
            { -0.7f, -0.8f, -0.9f },
            { -1.0f, -1.1f, -1.2f }
        }
    },

    // ===== Grupo 2 =====

    {   // out_channel 2
        {
            { 0.05f,  0.10f,  0.15f },
            { 0.20f,  0.25f,  0.30f }
        },
        {
            { -0.05f, -0.10f, -0.15f },
            { -0.20f, -0.25f, -0.30f }
        }
    },

    {   // out_channel 3
        {
            { 0.33f,  0.44f,  0.55f },
            { 0.66f,  0.77f,  0.88f }
        },
        {
            { -0.33f, -0.44f, -0.55f },
            { -0.66f, -0.77f, -0.88f }
        }
    }
};

float bias[4] = {0.5, -0.5, 0.25, -0.25};

float input[4][5][5] = {

    {   // Canal 0
        { 1.f, 2.f, 3.f, 4.f, 5.f },
        { 5.f, 4.f, 3.f, 2.f, 1.f },
        { 1.f, 1.f, 1.f, 1.f, 1.f },
        { 0.f, 0.f, 0.f, 0.f, 0.f },
        { 2.f, 2.f, 2.f, 2.f, 2.f }
    },

    {   // Canal 1
        { 0.f, 1.f, 0.f, 1.f, 0.f },
        { 1.f, 0.f, 1.f, 0.f, 1.f },
        { 0.f, 1.f, 0.f, 1.f, 0.f },
        { 1.f, 0.f, 1.f, 0.f, 1.f },
        { 0.f, 1.f, 0.f, 1.f, 0.f }
    },

    {   // Canal 2
        { 2.f, 2.f, 2.f, 2.f, 2.f },
        { 3.f, 3.f, 3.f, 3.f, 3.f },
        { 4.f, 4.f, 4.f, 4.f, 4.f },
        { 5.f, 5.f, 5.f, 5.f, 5.f },
        { 6.f, 6.f, 6.f, 6.f, 6.f }
    },

    {   // Canal 3
        { 9.f, 8.f, 7.f, 6.f, 5.f },
        { 4.f, 3.f, 2.f, 1.f, 0.f },
        { 1.f, 2.f, 3.f, 4.f, 5.f },
        { 6.f, 7.f, 8.f, 9.f, 0.f },
        { 1.f, 3.f, 5.f, 7.f, 9.f }
    }

};

int main()
{
    ofstream file("primitive_debug/conv2D/out_cpp.txt");

    float result[4][3][3];
    Conv2D<2, 3, 1, 1, 0, 0, 2, 1, 2>(input, result, weight, bias);

    file << "Sample 1:" << endl << endl;;
    
    for (int c = 0; c < 4; c++)
    {
        for (int h = 0; h < 3; h++)
        {
            for (int w = 0; w < 3; w++)
            {
                file << fixed << setprecision(4) << result[c][h][w] << " ";
            }
            
            file << endl;
        }
        
        file << endl;
    }

    return 0;
}