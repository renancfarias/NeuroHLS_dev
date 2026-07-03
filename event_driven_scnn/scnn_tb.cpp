#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm> 

#include "network.h" 
#include "types.h" 
#include "debug_utils.h"  

using namespace std;

// Caminhos dos arquivos (certifique-se que batem com o gerado pelo Python)
const string INPUT_FILE   = "n-mnist_test_150_steps_txt/nmnist_test_events_full.txt";
const string TARGET_FILE  = "n-mnist_test_150_steps_txt/nmnist_test_targets_full.txt";
const string OUTPUT_FILE  = "output_predictions.txt"; // Log de sumarização
const string OUTPUT_EVENTS_FILE = "output_events.txt"; // Log detalhado de CADA spike

const int NUM_CLASSES = 10; 

// CONFIGURAÇÃO: Número máximo de amostras para processar (-1 para todas)
const int MAX_SAMPLES = 2;
const int PRINT_SAMPLE_INTERVAL = 1;

int main() {
    // -------------------------------------------------------------------------
    // 1. STREAMS
    // -------------------------------------------------------------------------
// HLS Stream depth deve ser grande o suficiente para teste, mas em hardware é FIFO pequena.
    // HLS Stream depth deve ser grande o suficiente para teste, mas em hardware é FIFO pequena.
    hls::stream<spike_t> input_stream("input_stream");
    hls::stream<spike_t> output_stream("output_stream");

    // -------------------------------------------------------------------------
    // 2. CARREGAR TARGETS (Gabarito)
    // -------------------------------------------------------------------------
    vector<int> targets;
    ifstream target_in(TARGET_FILE);
    if (!target_in.is_open()) {
        cerr << "Erro: Nao foi possivel abrir targets: " << TARGET_FILE << endl;
        // Não retorna erro fatal, permite rodar sem acurácia se quiser
        // return 1; 
    } else {
        int lbl;
        while (target_in >> lbl) {
            targets.push_back(lbl);
        }
        target_in.close();
        cout << "Targets carregados: " << targets.size() << endl;
    }

    // -------------------------------------------------------------------------
    // 3. CARREGAR EVENTOS DE ENTRADA
    // -------------------------------------------------------------------------
    ifstream infile(INPUT_FILE);
    if (!infile.is_open()) {
        cerr << "Erro: Nao foi possivel abrir inputs: " << INPUT_FILE << endl;
        return 1;
    }

    // Variáveis temporárias para leitura do arquivo de texto
    // Formato assumido: type amp timestamp time_step batch channel height width
    int type_in, time_step_in, batch_in, ch_in, h_in, w_in;
    float amp_in, timestamp_in; 
    
    cout << "Processando eventos por STEP (alimentacao transacional do DUT)..." << endl;
    int max_batch_idx = -1;
    int total_events_loaded = 0;
    int total_steps_sent = 0;

    // -------------------------------------------------------------------------
    // 4. LOOP DE PROCESSAMENTO E INFERÊNCIA
    // -------------------------------------------------------------------------
    ofstream outfile(OUTPUT_FILE);
    outfile << "Sample_ID,True_Label,Predicted_Class,Spike_Count,Correct" << endl;

    // ARQUIVO DETALHADO DE EVENTOS
    ofstream event_logfile(OUTPUT_EVENTS_FILE);
    event_logfile << "type amplitude timestamp time_step batch_idx channel_idx height_idx width_idx" << endl;

    cout << "\nIniciando Inferência SCNN..." << endl;
if (MAX_SAMPLES > 0) cout << "Limitado a " << MAX_SAMPLES << " amostras." << endl;
    cout << "---------------------------------------------------------------" << endl;

    // Estado para contagem de votos (Rate Coding na saída)
    int spike_counts[NUM_CLASSES] = {0};
    int current_sample_id = 0;
    
    // Estatísticas Globais
    int total_samples_processed = 0;
    int correct_predictions = 0;

    // Leitura online do arquivo: envia 1 STEP por transação ao DUT
    while (infile >> type_in >> amp_in >> timestamp_in >> time_step_in >> batch_in >> ch_in >> h_in >> w_in) {

        if (MAX_SAMPLES > 0 && batch_in >= MAX_SAMPLES) {
            break;
        }

        spike_t s;
        s.type        = (int)type_in;
        s.amplitude   = (current_t)amp_in;
        s.timestamp   = (time_step_t)timestamp_in;
        s.time_step   = (u_int32_t)time_step_in;
        s.batch_idx   = (u_int32_t)batch_in;
        s.channel_idx = (u_int32_t)ch_in;
        s.height_idx  = (u_int32_t)h_in;
        s.width_idx   = (u_int32_t)w_in;

        input_stream.write(s);
        total_events_loaded++;
        max_batch_idx = batch_in;

        // Dispara DUT no final de cada STEP/SAMPLE para evitar sobra de vetor na co-sim
        if (s.type == TYPE_END_STEP || s.type == TYPE_END_SAMPLE) {
            total_steps_sent++;
            scnn(input_stream, output_stream);

            while (!output_stream.empty()) {
                spike_t out_s = output_stream.read();

                event_logfile << out_s.type << " "
                              << out_s.amplitude << " "
                              << out_s.timestamp << " "
                              << (int)out_s.time_step << " "
                              << out_s.batch_idx << " "
                              << out_s.channel_idx << " "
                              << out_s.height_idx << " "
                              << out_s.width_idx << endl;

                if (out_s.type == TYPE_SPIKE) {
                    int class_idx = out_s.width_idx;
                    if (class_idx >= 0 && class_idx < NUM_CLASSES) {
                        spike_counts[class_idx]++;
                    }
                }
                else if (out_s.type == TYPE_END_SAMPLE) {
                    int predicted_class = -1;
                    int max_spikes = -1;

                    for (int c = 0; c < NUM_CLASSES; c++) {
                        if (spike_counts[c] > max_spikes) {
                            max_spikes = spike_counts[c];
                            predicted_class = c;
                        }
                    }

                    if (max_spikes == 0) predicted_class = -1;

                    int true_label = -1;
                    bool is_correct = false;

                    if (current_sample_id < targets.size()) {
                        true_label = targets[current_sample_id];
                        if (predicted_class == true_label) {
                            correct_predictions++;
                            is_correct = true;
                        }
                    }

                    total_samples_processed++;

                    if (current_sample_id % PRINT_SAMPLE_INTERVAL == 0 || current_sample_id == 0) {
                        float acc = (float)correct_predictions / total_samples_processed * 100.0;
                        cout << "Sample " << setw(4) << current_sample_id
                             << " | Pred: " << predicted_class
                             << " | True: " << true_label
                             << " | Votes: " << max_spikes
                             << " | Acc Atual: " << setprecision(3) << acc << "%" << endl;
                    }

                    outfile << current_sample_id << ","
                            << true_label << ","
                            << predicted_class << ","
                            << max_spikes << ","
                            << (is_correct ? 1 : 0) << endl;

                    for (int c = 0; c < NUM_CLASSES; c++) spike_counts[c] = 0;
                    current_sample_id++;
                }

                spike_to_txt(out_s, "output_events_log.txt");
            }
        }
    }
    infile.close();

    // Flush de segurança para dados pendentes (ex.: arquivo sem token final)
    while (!input_stream.empty()) {
        total_steps_sent++;
        scnn(input_stream, output_stream);
        while (!output_stream.empty()) {
            spike_t out_s = output_stream.read();
            event_logfile << out_s.type << " "
                          << out_s.amplitude << " "
                          << out_s.timestamp << " "
                          << (int)out_s.time_step << " "
                          << out_s.batch_idx << " "
                          << out_s.channel_idx << " "
                          << out_s.height_idx << " "
                          << out_s.width_idx << endl;
            spike_to_txt(out_s, "output_events_log.txt");
        }
    }

    cout << "Total de eventos enviados: " << total_events_loaded << endl;
    cout << "Total de chamadas ao DUT (steps): " << total_steps_sent << endl;
    cout << "Ultimo Batch ID processado: " << max_batch_idx << endl;
    
    // -------------------------------------------------------------------------
    // 5. RESULTADO FINAL
    // -------------------------------------------------------------------------
    cout << "---------------------------------------------------------------" << endl;
    cout << "Simulacao Concluida." << endl;
    cout << "Total Samples: " << total_samples_processed << endl;
    cout << "Correct Preds: " << correct_predictions << endl;
    if (total_samples_processed > 0)
        cout << "FINAL ACCURACY: " << ((float)correct_predictions / total_samples_processed * 100.0) << "%" << endl;
    else
        cout << "Nenhuma amostra completa processada." << endl;

    outfile.close();
event_logfile.close();
    return 0;
}


