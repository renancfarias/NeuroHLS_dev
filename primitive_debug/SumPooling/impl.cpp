#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

/**
 * SumPooling HLS Paramétrico
 * --------------------------------------------
 * T:    Tipo de dado (int, float, etc)
 * IN_H: Altura da entrada
 * IN_W: Largura da entrada
 * K_H:  Altura do Kernel (Janela)
 * K_W:  Largura do Kernel (Janela)
 * S_H:  Stride Vertical (Passo Y)
 * S_W:  Stride Horizontal (Passo X)
 * P_H:  Padding Vertical (Adiciona zeros em Cima e Embaixo)
 * P_W:  Padding Horizontal (Adiciona zeros na Esquerda e Direita)
 */
template <
    typename T,
    int IN_H, int IN_W,
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W
>
void sum_pooling_custom(
    const T input[IN_H][IN_W],
    T output[(IN_H + 2*P_H - K_H) / S_H + 1][(IN_W + 2*P_W - K_W) / S_W + 1]
) {
    // Constantes de dimensão de saída
    const int OUT_H = (IN_H + 2 * P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - K_W) / S_W + 1;

    // Loop Vertical da Saída
    for (int i = 0; i < OUT_H; ++i) {
        
        // Loop Horizontal da Saída
        for (int j = 0; j < OUT_W; ++j) {
            
            T sum = 0;

            // --- Janela do Kernel (Retangular) ---
            
            // Loop Vertical do Kernel
            for (int ki = 0; ki < K_H; ++ki) {
                
                // Loop Horizontal do Kernel
                for (int kj = 0; kj < K_W; ++kj) {
                    
                    // Lógica de Endereçamento com Padding Virtual
                    int r_idx = (i * S_H) + ki - P_H;
                    int c_idx = (j * S_W) + kj - P_W;

                    // Verificação de Borda (Boundary Check)
                    if (r_idx >= 0 && r_idx < IN_H && c_idx >= 0 && c_idx < IN_W) {
                        sum += input[r_idx][c_idx];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// Wrapper para múltiplos canais
template <
    typename T,
    int CHANNELS,
    int IN_H, int IN_W,
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W
>
void sum_pooling_multi_channel(
    const T input[CHANNELS][IN_H][IN_W],
    T output[CHANNELS][(IN_H + 2*P_H - K_H) / S_H + 1][(IN_W + 2*P_W - K_W) / S_W + 1]
) {
    for (int c = 0; c < CHANNELS; ++c) {
        sum_pooling_custom<T, IN_H, IN_W, K_H, K_W, S_H, S_W, P_H, P_W>(
            input[c], output[c]
        );
    }
}

// ============================================================================
// TESTES
// ============================================================================

// Teste 1: 4x4 -> 2x2 (kernel 2x2, stride 2, padding 0)
float test1_input[4][4] = {
    {1.0, 2.0, 3.0, 4.0},
    {5.0, 6.0, 7.0, 8.0},
    {9.0, 10.0, 11.0, 12.0},
    {13.0, 14.0, 15.0, 16.0}
};

// Teste 2: 2 canais, 3x3 (kernel 2x2, stride 1, padding 0)
float test2_input[2][3][3] = {
    {{1.0, 2.0, 3.0},
     {4.0, 5.0, 6.0},
     {7.0, 8.0, 9.0}},
    
    {{0.5, 1.5, 2.5},
     {3.5, 4.5, 5.5},
     {6.5, 7.5, 8.5}}
};

// Teste 3: 3x3 com padding (kernel 2x2, stride 1, padding 1)
float test3_input[3][3] = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0},
    {7.0, 8.0, 9.0}
};

int main()
{
    ofstream file("primitive_debug/SumPooling/out_cpp.txt");
    
    cout << "============================================================" << endl;
    cout << "Teste 1: Input 4x4, Kernel 2x2, Stride 2, Padding 0" << endl;
    cout << "============================================================" << endl;
    
    // Output: 2x2
    float test1_output[2][2];
    sum_pooling_custom<float, 4, 4, 2, 2, 2, 2, 0, 0>(test1_input, test1_output);
    
    cout << "\nInput:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "  ";
        for (int j = 0; j < 4; j++) {
            cout << setw(6) << test1_input[i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nOutput:" << endl;
    for (int i = 0; i < 2; i++) {
        cout << "  ";
        for (int j = 0; j < 2; j++) {
            cout << setw(6) << test1_output[i][j] << " ";
        }
        cout << endl;
    }
    
    file << "Teste 1: 4x4 -> 2x2" << endl;
    file << "[[" << test1_output[0][0] << " " << test1_output[0][1] << "]" << endl;
    file << " [" << test1_output[1][0] << " " << test1_output[1][1] << "]]" << endl << endl;
    
    cout << "\nCálculo esperado:" << endl;
    cout << "  [0,0] = " << (1+2+5+6) << " (soma de 1,2,5,6)" << endl;
    cout << "  [0,1] = " << (3+4+7+8) << " (soma de 3,4,7,8)" << endl;
    cout << "  [1,0] = " << (9+10+13+14) << " (soma de 9,10,13,14)" << endl;
    cout << "  [1,1] = " << (11+12+15+16) << " (soma de 11,12,15,16)" << endl;
    
    // ========================================================================
    
    cout << "\n============================================================" << endl;
    cout << "Teste 2: Input 2ch 3x3, Kernel 2x2, Stride 1, Padding 0" << endl;
    cout << "============================================================" << endl;
    
    // Output: 2 canais, 2x2
    float test2_output[2][2][2];
    sum_pooling_multi_channel<float, 2, 3, 3, 2, 2, 1, 1, 0, 0>(test2_input, test2_output);
    
    cout << "\nCanal 0 Input:" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "  ";
        for (int j = 0; j < 3; j++) {
            cout << setw(6) << test2_input[0][i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nCanal 0 Output:" << endl;
    for (int i = 0; i < 2; i++) {
        cout << "  ";
        for (int j = 0; j < 2; j++) {
            cout << setw(6) << test2_output[0][i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nCanal 1 Input:" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "  ";
        for (int j = 0; j < 3; j++) {
            cout << setw(6) << test2_input[1][i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nCanal 1 Output:" << endl;
    for (int i = 0; i < 2; i++) {
        cout << "  ";
        for (int j = 0; j < 2; j++) {
            cout << setw(6) << test2_output[1][i][j] << " ";
        }
        cout << endl;
    }
    
    file << "Teste 2: 2ch 3x3 -> 2x2" << endl;
    file << "Canal 0:" << endl;
    file << "[[" << test2_output[0][0][0] << " " << test2_output[0][0][1] << "]" << endl;
    file << " [" << test2_output[0][1][0] << " " << test2_output[0][1][1] << "]]" << endl;
    file << "Canal 1:" << endl;
    file << "[[" << test2_output[1][0][0] << " " << test2_output[1][0][1] << "]" << endl;
    file << " [" << test2_output[1][1][0] << " " << test2_output[1][1][1] << "]]" << endl << endl;
    
    // ========================================================================
    
    cout << "\n============================================================" << endl;
    cout << "Teste 3: Input 3x3, Kernel 2x2, Stride 1, Padding 1" << endl;
    cout << "============================================================" << endl;
    
    // Output: 4x4 (com padding, 3+2*1=5, (5-2)/1+1=4)
    float test3_output[4][4];
    sum_pooling_custom<float, 3, 3, 2, 2, 1, 1, 1, 1>(test3_input, test3_output);
    
    cout << "\nInput:" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "  ";
        for (int j = 0; j < 3; j++) {
            cout << setw(6) << test3_input[i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nOutput:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "  ";
        for (int j = 0; j < 4; j++) {
            cout << setw(6) << test3_output[i][j] << " ";
        }
        cout << endl;
    }
    
    file << "Teste 3: 3x3 com padding -> 4x4" << endl;
    for (int i = 0; i < 4; i++) {
        file << "[";
        for (int j = 0; j < 4; j++) {
            file << test3_output[i][j];
            if (j < 3) file << " ";
        }
        file << "]" << endl;
    }
    
    cout << "\n============================================================" << endl;
    cout << "Resultados salvos em: primitive_debug/SumPooling/out_cpp.txt" << endl;
    cout << "============================================================" << endl;
    
    file.close();
    
    return 0;
}
