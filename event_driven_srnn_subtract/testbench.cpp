#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "hls_stream.h"

#include "types.h"
#include "quantization.h"
#include "snn_implementation.h"

using namespace std;

#define TOTAL_SAMPLES 140
#define BATCH_SIZE 5
#define DIM_1 12
#define OUTPUT_SIZE 7
#define STEP_COUNT 256

int main (int argc, char **argv)
{
    string data_file_name = "tb_data/data.txt";
    string targets_file_name = "tb_data/targets.txt";

    float dt = 1e-4;

    int packet_id = 0;

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

    // Assuming input_t is defined in quantization.h or types.h
    // and correctly parses from input_file
    int input_data[BATCH_SIZE][STEP_COUNT][DIM_1];
    int target_data[BATCH_SIZE];

    hls::stream<spike_t> input_stream("input_stream");
    hls::stream<spike_t> output_stream("output_stream");

    int total_correct = 0;
    int total_batches = TOTAL_SAMPLES / BATCH_SIZE;

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
                    double val;
                    input_file >> val; // Reading safely
                    input_data[b][s][d1] = (int)val;
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


            

            // Encode sample into spike packets
            for (int s = 0; s < STEP_COUNT; s++)
            {
                for (int d1 = 0; d1 < DIM_1; d1++)
                {
                    if (input_data[b][s][d1] > 0) // if there is a spike
                    {
                        spike_t spk;
                        spk.type = TYPE_SPIKE;
                        spk.amplitude = input_data[b][s][d1];
                        spk.timestamp = (float)s*dt;
                        spk.time_step = (ap_uint<16>)s;
                        spk.batch_idx = (ap_uint<16>)b;
                        spk.channel_idx = packet_id++;
                        spk.height_idx = 0;
                        spk.width_idx = (ap_uint<16>)d1;
                        input_stream.write(spk);
                    }
                }
                // End of Step packet
                spike_t end_step_spk;
                end_step_spk.type = TYPE_END_STEP;
                end_step_spk.timestamp = (float)s*dt;
                end_step_spk.time_step = s;
                end_step_spk.channel_idx = packet_id++;
                input_stream.write(end_step_spk);
            }

            // End of Sample packet
            spike_t end_sample_spk;
            end_sample_spk.type = TYPE_END_SAMPLE;
            end_sample_spk.timestamp = (float)STEP_COUNT*dt;
            end_sample_spk.time_step = STEP_COUNT;
            end_sample_spk.channel_idx = packet_id++;
            input_stream.write(end_sample_spk);



            // Run the top function
            while (!input_stream.empty())
            {
                snn_to_hls(input_stream, output_stream);
            }

            // Process output stream
            accum_t accum_output[OUTPUT_SIZE] = {};

            while (!output_stream.empty())
            {
                spike_t out_spk = output_stream.read();
                
                if (out_spk.type == TYPE_SPIKE)
                {
                    int out_idx = out_spk.width_idx; // or channel_idx depending on your output configuration
                    if (out_idx >= 0 && out_idx < OUTPUT_SIZE)
                    {
                        accum_output[out_idx] += out_spk.amplitude;
                    }
                }
            }

            // Find max spike count
            accum_t max_v = -1;
            int idx_max = 0;

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                if (accum_output[i] > max_v)
                {
                    max_v = accum_output[i];
                    idx_max = i;
                }
            }

            cout << "  Sample (batch=" << cur_batch + 1 << ", b=" << b << ")"
                 << "  predicted=" << idx_max
                 << "  expected=" << target_data[b]
                 << (idx_max == target_data[b] ? "  [OK]" : "  [FAIL]") << endl;

            if (idx_max == target_data[b])
            {
                batch_total_correct++;
            }
        }

        cout << "Batch (" << cur_batch + 1 << " / " << total_batches << "): ";
        cout << fixed << setprecision(2) << (float) 100 * batch_total_correct / BATCH_SIZE << "%" << endl;
        total_correct += batch_total_correct;
    }

    cout << endl << " *** Final Acc: " << (float) 100 * total_correct / TOTAL_SAMPLES << "%" << endl;
    cout << "-------------------------------" << endl << endl;

    cout << "input_stream vazio ao final do testbench: "
    << (input_stream.empty() ? "sim" : "nao") << endl;

    input_file.close();
    targets_file.close();

    return 0;
}
