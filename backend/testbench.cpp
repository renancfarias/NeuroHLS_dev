#include <deque>
#include <iostream>
#include <ostream>
#include <vector>

#include "types_and_params.h"

using namespace std;

#define BATCH_SIZE_TEST 24 // POR ENQUANTO, SOMENTE FUNCIONA COM DIVISORES DO TOTAL DE AMOSTRAS

int main (int argc, char **argv)
{
    string log_file_name = "log_testbench.txt";
    string data_file_name = "data.txt";
    string targets_file_name = "targets.txt";

    std::ifstream input_file(data_file_name);
    if (!input_file.is_open())
    {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    std::ifstream targets_file(targets_file_name);
    if (!targets_file.is_open())
    {
        cerr << "Error opening targets file." << endl;
        return 1;
    }

    ofstream tb_log(log_file_name, ios::trunc);

    //<decl_input_data>
    int target_data[BATCH_SIZE_TEST];

    int total_correct = 0;
    int total_batches = (float)TOTAL_SAMPLES / BATCH_SIZE_TEST;

    for (int cur_batch = 0; cur_batch < total_batches; cur_batch++)
    {
        for (int b = 0; b < BATCH_SIZE_TEST; b++)
        {
            //<read_batch>
        }
        
        for (int b = 0; b < BATCH_SIZE_TEST; b++)
        {
            targets_file >> target_data[b];
        }

        int batch_total_correct = 0;

        for (int b = 0; b < BATCH_SIZE_TEST; b++)
        {
            int accum_output[OUTPUT_SIZE] = {};
            bit_t output[OUTPUT_SIZE];

            for (int s = 0; s < STEP_COUNT; s++)
            {
                //<feed_data_snn>

                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    accum_output[i] += output[i];
                }
            }

            int max_v = -1;
            int idx_max = 0;

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                if (accum_output[i] > max_v)
                {
                    max_v = accum_output[i];
                    idx_max = i;
                }
            }

            if (idx_max == target_data[b])
            {
                batch_total_correct++;
            }
        }

        tb_log << "Batch (" << cur_batch + 1 << " / " << total_batches << ") :" << endl;

        tb_log << fixed << setprecision(2) << "Acc: " << (float) 100 * batch_total_correct / BATCH_SIZE_TEST << "%" << endl;
        total_correct += batch_total_correct;
    }

    tb_log << "Final Acc: " << (float) 100 * total_correct / TOTAL_SAMPLES << "%" << endl;

    input_file.close();
    targets_file.close();
    tb_log.close();

    return 0;
}