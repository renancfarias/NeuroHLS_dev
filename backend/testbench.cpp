#include <deque>
#include <iostream>
#include <ostream>
#include <vector>

using namespace std;

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