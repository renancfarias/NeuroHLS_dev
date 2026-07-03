#include <deque>
#include <fstream>
#include <iostream>
#include <vector>
#include "network.h"
#include "types.h"
#include "lif.h"

const int PACKAGE = 100;
const int NUM_STEPS = 10;
const int NUM_SAMPLES = 1000;

using namespace std;


int main (int argc, char **argv)
{   
    // load inputs from txt file
    current_t input_data[PACKAGE][NUM_STEPS][NUM_INPUTS];

    hls::stream<spike_t, 784> input_stream("input_stream");
    hls::stream<spike_t, 10> output_stream("output_stream");

    int target_data[PACKAGE];

    time_step_t time = 0.0;

    int output_spike_count[NUM_OUTPUTS] = {0};
    
    int total_correct=0;
    
    // open file with inputs
    std::ifstream input_file("n-mnist_testset_data.txt");
    if (!input_file.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    // open file with targets
    std::ifstream targets_file("n-mnist_testset_targets.txt");
    if (!targets_file.is_open()) {
        std::cerr << "Error opening targets file." << std::endl;
        return 1;
    }

    const int total_batch= NUM_SAMPLES/PACKAGE;
    cout<<"Number of batch to process:"<<total_batch<<endl;
    for(int number_batch=0;number_batch<total_batch;number_batch++){

        cout<<"BATCH number: "<< (number_batch+1)<<endl;
        
        for (int i = 0; i < PACKAGE; i++) {
            for (int j = 0; j < NUM_STEPS; j++) {
                for (int k = 0; k < NUM_INPUTS; k++) {
                    input_file >> input_data[i][j][k];
                }
            }
        }


        for (int i = 0; i < PACKAGE; i++) {
            targets_file >> target_data[i];
        }

        int output_data[NUM_SAMPLES];

        for (int i = 0; i < PACKAGE; i++)
        {

            for (int j = 0; j < NUM_STEPS; j++)
            {
                
                get_spikes<784>(input_data[i][j], input_stream, time);
                
                snn(input_stream, output_stream);
                
                // read output spikes
                while (output_stream.empty() == false)
                {
                    spike_t out_spike = output_stream.read();
                    output_spike_count[out_spike.index] += (int)out_spike.amplitude;
                }
                time += DT;
            } 

            // find the index of the neuron with the highest spike count
            int max_index = 0;
            int max_value = output_spike_count[0];
            for (int j = 1; j < NUM_OUTPUTS; j++) {
                if (output_spike_count[j] > max_value) {
                    max_value = output_spike_count[j];
                    max_index = j;
                }
            }
            output_data[i] = max_index; 
            // reset spike counts for next sample
            for (int j = 0; j < NUM_OUTPUTS; j++) {
                output_spike_count[j] = 0;
            } 
                      
        }


        // calculate the accuracy
        int correct = 0;
        for (int i = 0; i < PACKAGE; i++) {
            if (output_data[i] == target_data[i]) {
                correct++;
                total_correct++;
            }
        }
        float accuracy_batch = (float) correct / PACKAGE * 100.0;

        cout << "--------------" << endl;
        cout << "Result Batch "<<(number_batch+1)<<"/"<< total_batch << endl;
        cout << "Total Correct Batch: " << correct << endl;
        cout << "Total Tested Batch: " << PACKAGE << endl << endl;
        cout << "Accuracy Batch: " << accuracy_batch << "%" << endl;
        cout << "--------------" << endl;

    }
    float accuracy = (float) total_correct / NUM_SAMPLES * 100.0;

    cout << "--------------" << endl;
    cout << "Total Correct: " << total_correct << endl;
    cout << "Total Tested: " << NUM_SAMPLES << endl << endl;

    cout << "Accuracy: " << accuracy << "%" << endl;
    cout << "--------------" << endl;
    targets_file.close();
    input_file.close();

    return 0;
}