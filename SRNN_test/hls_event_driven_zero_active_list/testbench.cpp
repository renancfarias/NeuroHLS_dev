#include <iostream>
#include <iomanip>
#include <ostream>
#include <fstream>
#include <vector>

#include "quantization.h"
#include "snn_implementation.h"

using namespace std;

#define TOTAL_SAMPLES 7
#define BATCH_SIZE 7
#define DIM_1 12
#define OUTPUT_SIZE 7
#define STEP_COUNT 256


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

    static input_t input_data[BATCH_SIZE][STEP_COUNT][DIM_1];
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
					input_file >> input_data[b][s][d1];
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
            
            hls::stream<ed_spike_t> input_stream;
            hls::stream<ed_spike_t> output_stream;

            for (int s = 0; s < STEP_COUNT; s++)
            {
                {
    ed_spike_t event = {};

    if (input_data[b][s][0] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][0];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 0;
        input_stream.write(event);
    }
    if (input_data[b][s][1] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][1];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 1;
        input_stream.write(event);
    }
    if (input_data[b][s][2] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][2];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 2;
        input_stream.write(event);
    }
    if (input_data[b][s][3] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][3];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 3;
        input_stream.write(event);
    }
    if (input_data[b][s][4] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][4];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 4;
        input_stream.write(event);
    }
    if (input_data[b][s][5] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][5];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 5;
        input_stream.write(event);
    }
    if (input_data[b][s][6] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][6];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 6;
        input_stream.write(event);
    }
    if (input_data[b][s][7] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][7];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 7;
        input_stream.write(event);
    }
    if (input_data[b][s][8] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][8];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 8;
        input_stream.write(event);
    }
    if (input_data[b][s][9] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][9];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 9;
        input_stream.write(event);
    }
    if (input_data[b][s][10] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][10];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 10;
        input_stream.write(event);
    }
    if (input_data[b][s][11] != input_t(0)) {
        event = {};
        event.type = ED_TYPE_SPIKE;
        event.amplitude = input_data[b][s][11];
        event.timestamp = (ed_time_step_t)(s * NEURO_HLS_EVENT_DT);
        event.time_step = s;
        event.channel_idx = 0;
        event.height_idx = 0;
        event.width_idx = 11;
        input_stream.write(event);
    }
    event = {};
    event.type = s == STEP_COUNT - 1 ? ED_TYPE_END_SAMPLE : ED_TYPE_END_STEP;
    event.timestamp = (ed_time_step_t)((s + 1) * NEURO_HLS_EVENT_DT);
    event.time_step = s;
    input_stream.write(event);
    snn_to_hls(input_stream, output_stream, s == 0);
}

                while (true) {
                    ed_spike_t event = output_stream.read();
                    if (event.type == ED_TYPE_SPIKE && event.width_idx < OUTPUT_SIZE)
                        accum_output[event.width_idx]++;
                    if (event.type != ED_TYPE_SPIKE) break;
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
