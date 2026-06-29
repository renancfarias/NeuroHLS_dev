#include <iostream>
#include <ostream>
#include <fstream>
#include <vector>

#include "quantization.h"
#include "snn_implementation.h"

using namespace std;

#define TOTAL_SAMPLES 1024
#define BATCH_SIZE 2
#define DIM_1 2
#define DIM_2 34
#define DIM_3 34
#define OUTPUT_SIZE 10
#define STEP_COUNT 300


int main (int argc, char **argv)
{
    string data_file_name = "data.txt";
    string targets_file_name = "targets.txt";

    ifstream input_file(data_file_name);
    if (!input_file.is_open())
    {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    ifstream targets_file(targets_file_name);
    if (!targets_file.is_open())
    {
        cerr << "Error opening targets file." << endl;
        return 1;
    }

    input_t input_data[BATCH_SIZE][STEP_COUNT][DIM_1][DIM_2][DIM_3];
    int target_data[BATCH_SIZE];

    int total_correct = 0;
    int total_batches = (float)TOTAL_SAMPLES / BATCH_SIZE;

    cout << endl << "-------------------------------" << endl;
    cout << " - Total Samples: " << TOTAL_SAMPLES << endl;
    cout << " - Batch Size: " << BATCH_SIZE << endl;
    cout << "-------------------------------" << endl << endl;

    for (int cur_batch = 0; cur_batch < total_batches; cur_batch++)
    {
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            for (int s = 0; s < STEP_COUNT; s++)
			{
				for (int d1 = 0; d1 < DIM_1; d1++)
				{
					for (int d2 = 0; d2 < DIM_2; d2++)
					{
						for (int d3 = 0; d3 < DIM_3; d3++)
						{
							input_file >> input_data[b][s][d1][d2][d3];
						}
					}
				}
			}
        }
        
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            targets_file >> target_data[b];
        }

        int batch_total_correct = 0;

        for (int b = 0; b < BATCH_SIZE; b++)
        {
            int accum_output[OUTPUT_SIZE] = {};
            bit_t output[OUTPUT_SIZE];

            for (int s = 0; s < STEP_COUNT; s++)
            {
                snn_to_hls(input_data[b][s], output, s == 0);

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

            //<debug_output>
        }

        cout << "Batch (" << cur_batch + 1 << " / " << total_batches << "): ";
        cout << fixed << setprecision(2) << (float) 100 * batch_total_correct / BATCH_SIZE << "%" << endl;
        
        total_correct += batch_total_correct;
    }

    cout << endl << " *** Final Acc: " << (float) 100 * total_correct / TOTAL_SAMPLES << "%" << endl;
    cout << "-------------------------------" << endl << endl;

    input_file.close();
    targets_file.close();

    return 0;
}