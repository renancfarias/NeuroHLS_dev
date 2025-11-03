#include <deque>
#include <iostream>
#include <ostream>
#include <vector>
#include "layer.h"

// #include "ecg_ptbdb/weights.h"

// void test_conv_1d()
// {
//     float inputs[2][7] = {{1, 2, 3, 4, 5, 6, 7},
//                         {7, 6, 5, 4, 3, 2, 1}};

//     float filter[3][2][3] = {
//     {{1, 0, -1},
//     {0.5, 0, -0.5}},

//     {{1, 1, 1},
//     {0, 0, 0}},

//     {{0, 0, 0},
//     {1, -1, 1}}
//     };

//     //conv_1d<2, 7, 3, 3, 1, float>(inputs, filter);
// }

// void test_2_conv_1d()
// {
//     float inputs[3][10] = 
//     {
//         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},       
//         {10, 9, 8, 7, 6, 5, 4, 3, 2, 1},     
//         {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
//     };

//     float filter[2][3][4] = 
//     {
//         {
//             {1.0, 0.0, 0.0, -1.0},
//             {0.5, 0.5, 0.5, 0.5},
//             {0.0, 1.0, -1.0, 0.0}
//         },
        
//         {
//             {0.0, 1.0, 1.0, 0.0},
//             {1.0, -1.0, 0.0, 0.0},
//             {0.5, 0.5, 0.5, 0.5}
//         }
//     };


//     //conv_1d<3, 10, 2, 4, 1, float>(inputs, filter);
// }

// void test_max_pool_2d()
// {
//     int input[1][4][4] = 
//     {   
//         {
//             {1, 2, 3, 4},
//             {5, 6, 7, 8},
//             {9, 10, 11, 12},
//             {13, 14, 15, 16}
//         }
//     };

//     int result[1][2][2];
//     max_pool_2d<1, 4, 4, 2, 2, int, int>(input, result);

//     for (int i = 0; i < 2; i++)
//     {
//         for (int j = 0; j < 2; j++)
//         {
//             cout << result[0][i][j] << " ";
//         }

//         cout << endl;
//     }
// }

// template<int n_neurons, int n_inputs>
// void print_info_weights(weight_t w[n_neurons][n_inputs])
// {
//     float min_v = 1000;
//     float max_v = -1;

//     for (int n = 0; n < n_neurons; n++)
//     {
//         for (int i = 0; i < n_inputs; i++)
//         {
//             if (w[n][i] < min_v)
//             {
//                 min_v = w[n][i];
//             }

//             if (w[n][i] > max_v)
//             {
//                 max_v = w[n][i]; 
//             }
//         }
//     }

//     cout << "Info W[" << n_neurons << "][" << n_inputs << "]:" << endl;
//     cout << "Min: " << min_v << endl;
//     cout << "Max: " << max_v << endl;
// }

// void test_conv_1d_hls()
// {
//     float inputs[2][7] = {{1, 2, 3, 4, 5, 6, 7},
//                         {7, 6, 5, 4, 3, 2, 1}};

//     float result[3][5] = {};

//     snn_mnist_hls(inputs, result);

//     for (int i = 0; i < 3; i++)
//     {
//         for (int j = 0; j < 5; j++)
//         {
//             cout << result[i][j] << " ";
//         }
        
//         cout << endl;
//     }
// }

// void compare_conv_2d()
// {

// }

// void test_conv_2d()
// {
//     int input[2][4][4] = 
//     {
//         {
//             {1, 2, 3, 4},
//             {5, 6, 7, 8},
//             {9, 8, 7, 6},
//             {5, 4, 3, 2}
//         },

//         {
//             {2, 1, 0, -1},
//             {0, 1, 2, 3},
//             {3, 2, 1, 0},
//             {1, 0, -1, -2}
//         }
//     };

//     int filter[3][2][2][2] = 
//     {
//         {
//             {
//                 {1, 0},
//                 {0, -1}
//             },

//             {
//                 {0, 1},
//                 {-1, 0}
//             }
//         },

//         {
//             {
//                 {1, 1},
//                 {1, 1}
//             },

//             {
//                 {-1, -1},
//                 {-1, -1}
//             }
//         },

//         {
//             {
//                 {0, 1},
//                 {1, 0}
//             },

//             {
//                 {1, 0},
//                 {0, -1}
//             }
//         }
//     };
    
//     int result[3][3][3] = {};

//     conv_2d<4, 4, 2, 2, 2, 3, 1, int, int, int>(input, filter, result);

//     for (int ch = 0; ch < 3; ch++)
//     {
//         cout << "Channel " << ch + 1 << ":" << endl << endl;

//         for (int l = 0; l < 3; l++)
//         {
//             for (int c = 0; c < 3; c++)
//             {
//                 cout << result[ch][l][c] << " ";
//             }

//             cout << endl;
//         }

//         cout << "---------------" << endl << endl;
//     }
// }

// void test_conv_2d_2()
// {
//     int input[3][5][6] =
//     {
//         {   // canal 0
//             { 1, 2, 3, 4, 5, 6 },
//             { 6, 5, 4, 3, 2, 1 },
//             { 1, 1, 1, 1, 1, 1 },
//             { 2, 2, 2, 2, 2, 2 },
//             { 3, 3, 3, 3, 3, 3 }
//         },
//         {   // canal 1
//             { 0, 1, 2, 3, 4, 5 },
//             { 5, 4, 3, 2, 1, 0 },
//             { 1, 2, 3, 4, 5, 6 },
//             { 6, 5, 4, 3, 2, 1 },
//             { 0, 0, 0, 0, 0, 0 }
//         },
//         {   // canal 2
//             { 1, 0, 1, 0, 1, 0 },
//             { 0, 1, 0, 1, 0, 1 },
//             { 1, 0, 1, 0, 1, 0 },
//             { 0, 1, 0, 1, 0, 1 },
//             { 1, 0, 1, 0, 1, 0 }
//         }
//     };

//     int filter[2][3][2][3] =
//     {
//         {   // filtro 0
//             { 
//                 {  1,  0, -1 },
//                 {  0,  1,  0 } 
//             },

//             {
//                 {  0,  1,  0 },
//                 {  1,  0, -1 }
//             },

//             {
//                 { -1,  0,  1 },
//                 {  0, -1,  0 }
//             }
//         },

//         {   // filtro 1
//             {
//                 {  1,  1,  1 },
//                 {  1,  1,  1 }
//             },

//             {
//                 { -1, -1, -1 },
//                 { -1, -1, -1 } 
//             },

//             {   
//                 {  0,  0,  0 },
//                 {  0,  0,  0 }
//             }
//         }
//     };

//     const int out_ch = 2;
//     const int out_h = 4;
//     const int out_w = 4;
    
//     int result[out_ch][out_h][out_w] = {};

//     conv_2d<5, 6, 2, 3, 3, 2, 1, int, int, int>(input, filter, result);

//     for (int ch = 0; ch < out_ch; ch++)
//     {
//         cout << "Channel " << ch + 1 << ":" << endl << endl;

//         for (int l = 0; l < out_h; l++)
//         {
//             for (int c = 0; c < out_w; c++)
//             {
//                 cout << result[ch][l][c] << " ";
//             }

//             cout << endl;
//         }

//         cout << "---------------" << endl << endl;
//     }
// }

using namespace std;

// int main (int argc, char **argv)
// {
    // -----------------------------  TESTE FUNCIONAMENTO SCNN ----------------------
    
    // print_info_weights<num_neurons[0], NUM_INPUTS>(weights_0);
    // print_info_weights<num_neurons[1], num_neurons[0]>(weights_1);

    //test_conv_1d();

    // test_max_pool_2d();

    // test_conv_2d();

    // test_conv_2d();

    // -------------------------------------------------------------------------------

    // input_ecg_t input_data[NUM_SAMPLES][NUM_INPUTS];

    // std::ifstream input_file("inputs.txt");

    // if (!input_file.is_open())
    // {
    //     std::cerr << "Error opening input file." << std::endl;
    //     return 1;
    // }

    // for (int i = 0; i < NUM_SAMPLES; i++)
    // {
    //     for (int k = 0; k < NUM_INPUTS; k++)
    //     {
    //         input_file >> input_data[i][k];
    //     }
    // }

    // input_file.close();

    // int target_data[NUM_SAMPLES];

    // std::ifstream targets_file("targets.txt");

    // if (!targets_file.is_open())
    // {
    //     std::cerr << "Error opening targets file." << std::endl;
    //     return 1;
    // }

    // for (int i = 0; i < NUM_SAMPLES; i++)
    // {
    //     targets_file >> target_data[i];
    // }

    // targets_file.close();

    // cout << "Starting Inferences..." << endl;

    // int output_data[NUM_SAMPLES];

    // for (int i = 0; i < NUM_SAMPLES; i++)
    // {
    //     bit_t output [NUM_STEPS][num_neurons[1]];

    //     cout << "Input data: ";
    //     for(int k = 0; k < 10; k++)
    //     {
    //         cout << fixed << setprecision(4) << input_data[0][k] << " ";
    //     }

    //     cout << endl;

    //     for (int j = 0; j < NUM_STEPS; j++)
    //     {
    //         cout << j + 1 << endl;
    //         snn_mnist_hls(input_data[i], output[j]);
    //         // cout << output[j][0] << " " << output[j][1] << endl;
    //     }

    //     // cout << "spikes:" << endl;
    //     // for (int j = 0; j < NUM_STEPS; j++)
    //     // {
    //     //     cout << j << " : " << output[j][0] << " " << output[j][1] << endl;
    //     //     //snn_mnist_hls(input_data[i], output[j]);
    //     // }

    //     cout << "Tested: " << i << endl;

    //     // if (i % 50 == 0)
    //     // {
    //     //     cout << "Tested: " << i << endl;
    //     // }
        
    //     int count [num_neurons[1]] = {0};
    //     for (int j = 0; j < NUM_STEPS; j++)
    //     {
    //         for (int k = 0; k < num_neurons[1]; k++)
    //         {
    //             if (output[j][k] > 0)
    //             {
    //                 count[k]++;
    //             }
    //         }
    //     }

    //     int max_count = 0;
    //     int max_index = 0;
    //     for (int j = 0; j < num_neurons[1]; j++)
    //     {
    //         if (count[j] > max_count)
    //         {
    //             max_count = count[j];
    //             max_index = j;
    //         }
    //     }

    //     output_data[i] = max_index;

    //     // cout << "Teste " << i << ": ";

    //     // if (output_data[i] == target_data[i])
    //     // {
    //     //     cout << "CORRETO" << endl;
    //     // }
    //     // else
    //     // {
    //     //     cout << "ERRADO (estimado: " << output_data[i] << " - correto: " << target_data[i] << ")" << endl;
    //     // }
    // }

    // int correct = 0;
    // for (int i = 0; i < NUM_SAMPLES; i++)
    // {
    //     if (output_data[i] == target_data[i])
    //     {
    //         correct++;
    //     }
    // }
    
    // float accuracy = (float) correct / NUM_SAMPLES * 100.0;

    // cout << "--------------" << endl;
    // cout << "Total Correct: " << correct << endl;
    // cout << "Total Tested: " << NUM_SAMPLES << endl << endl;

    // cout << "Accuracy: " << accuracy << "%" << endl;
    // cout << "--------------" << endl;

    // return 0;
// }

// int main (int argc, char **argv)
// {   
//     // load inputs from txt file
//     bit_t input_data[NUM_SAMPLES][NUM_STEPS][NUM_INPUTS];

//     // open file with inputs
//     std::ifstream input_file("n-mnist_testset_data.txt");
//     if (!input_file.is_open()) {
//         std::cerr << "Error opening input file." << std::endl;
//         return 1;
//     }

//     for (int i = 0; i < NUM_SAMPLES; i++) {
//         for (int j = 0; j < NUM_STEPS; j++) {
//             for (int k = 0; k < NUM_INPUTS; k++) {
//                 input_file >> input_data[i][j][k];
//             }
//         }
//     }

//     // close the input file
//     input_file.close();

//     int target_data[NUM_SAMPLES];

//     // open file with targets
//     std::ifstream targets_file("n-mnist_testset_targets.txt");
//     if (!targets_file.is_open()) {
//         std::cerr << "Error opening targets file." << std::endl;
//         return 1;
//     }
//     for (int i = 0; i < NUM_SAMPLES; i++) {
//         targets_file >> target_data[i];
//     }
//     targets_file.close();

//     int output_data[NUM_SAMPLES];

//     for (int i = 0; i < NUM_SAMPLES; i++)
//     {
//         bit_t output [NUM_STEPS][num_neurons[1]];
//         bit_t step_input[NUM_INPUTS];
//         // hls::stream<neuron_idx_t> spike_index_stream;

//         for (int j = 0; j < NUM_STEPS; j++)
//         {
//             for (int k = 0; k < NUM_INPUTS; k++)
//             {
//                 if (input_data[i][j][k] > 0.0)
//                 {
//                     step_input[k] = 1;                    
//                     // spike_index_stream.write(k);
//                 }
//                 else
//                 {
//                     step_input[k] = 0;
//                 }
//             }

//             snn_mnist_hls(step_input, output[j]);
//             // snn_mnist_hls(spike_index_stream, output[j]);
//         } 
        
//         int count [num_neurons[1]] = {0};
//         for (int j = 0; j < NUM_STEPS; j++)
//         {
//             for (int k = 0; k < num_neurons[1]; k++)
//             {
//                 if (output[j][k] > 0)
//                 {
//                     count[k]++;
//                 }
//             }
//         }

//         // find the neuron with the maximum count
//         int max_count = 0;
//         int max_index = 0;
//         for (int j = 0; j < num_neurons[1]; j++) {
//             if (count[j] > max_count) {
//                 max_count = count[j];
//                 max_index = j;
//             }
//         }

//         output_data[i] = max_index;
//     }


//     // calculate the accuracy
//     int correct = 0;
//     for (int i = 0; i < NUM_SAMPLES; i++) {
//         if (output_data[i] == target_data[i]) {
//             correct++;
//         }
//     }
    
//     float accuracy = (float) correct / NUM_SAMPLES * 100.0;

//     cout << "--------------" << endl;
//     cout << "Total Correct: " << correct << endl;
//     cout << "Total Tested: " << NUM_SAMPLES << endl << endl;

//     cout << "Accuracy: " << accuracy << "%" << endl;
//     cout << "--------------" << endl;

//     return 0;
// }

void clear_debug_net()
{
    ofstream dnet("debug_net_VITIS.txt", ios::trunc);
    dnet.close();
}

#define BATCH_SIZE_TEST 24 // POR ENQUANTO, SOMENTE FUNCIONA COM DIVISORES DO TOTAL DE AMOSTRAS
#define IMAGE_DIM 28

int main (int argc, char **argv)
{
    clear_debug_net();

    ofstream dnet("debug_net_VITIS.txt", ios::trunc);

    // load inputs from txt file
    input_t input_data[BATCH_SIZE_TEST][IMAGE_DIM][IMAGE_DIM];
    int target_data[BATCH_SIZE_TEST];

    std::ifstream input_file("data.txt");
    if (!input_file.is_open())
    {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    std::ifstream targets_file("targets.txt");
    if (!targets_file.is_open())
    {
        std::cerr << "Error opening targets file." << std::endl;
        return 1;
    }

    int total_correct = 0;
    int total_batches = (float)TOTAL_SAMPLES / BATCH_SIZE_TEST;

    for (int cur_batch = 0; cur_batch < total_batches; cur_batch++)
    {
        //cout << "Current Batch: " << cur_batch << endl;
        
        for (int b = 0; b < BATCH_SIZE_TEST; b++)
        {
            for (int l = 0; l < IMAGE_DIM; l++)
            {
                for (int c = 0; c < IMAGE_DIM; c++)
                {
                    input_file >> input_data[b][l][c];
                }
            }
        }
        
        for (int b = 0; b < BATCH_SIZE_TEST; b++)
        {
            targets_file >> target_data[b];
        }

        int batch_total_correct = 0;

        for (int b = 0; b < BATCH_SIZE_TEST; b++)
        {
            //cout << "b = " << b << endl;

            int accum_output[NUM_OUTPUTS] = {};
            bit_t output[NUM_OUTPUTS];

            //print_mat<1>(input_data, "imagem-input");

            for (int s = 0; s < NUM_STEPS; s++)
            {
                snn_mnist_hls(input_data[b], output);

                for (int i = 0; i < NUM_OUTPUTS; i++)
                {
                    accum_output[i] += output[i];
                }

                return 0;
            }

            //cout << endl;
            //print_vet<10>(accum_output, "accum spikes");
            //return 0;

            int max_v = -1;
            int idx_max = 0;

            for (int i = 0; i < NUM_OUTPUTS; i++)
            {
                if (accum_output[i] > max_v)
                {
                    max_v = accum_output[i];
                    idx_max = i;
                }
            }

            //cout << "Inf: " << idx_max << " (Target: " << target_data[b] <<  ")" << endl;

            if (idx_max == target_data[b])
            {
                batch_total_correct++;
            }
        }

        dnet << "Batch (" << cur_batch + 1 << " / " << total_batches << ") :" << endl;

        dnet << fixed << setprecision(2) << "Acc: " << (float) 100 * batch_total_correct / BATCH_SIZE_TEST << "%" << endl;
        total_correct += batch_total_correct;
    }

    dnet << "Final Acc: " << (float) 100 * total_correct / TOTAL_SAMPLES << "%" << endl;

    input_file.close();
    targets_file.close();

    dnet.close();

    return 0;
}