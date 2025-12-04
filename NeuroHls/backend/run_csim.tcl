# ==================================
# Script para Vitis HLS: C Simulation
# ==================================

# Cria projeto
open_project -reset projeto_teste_26_nov

# Define a função top (essa é a função que será sintetizada no futuro)
set_top snn_to_hls

# Adiciona arquivos do projeto e o testbench

add_files snn_implementation.cpp
add_files snn_implementation.h
add_files weights.h
add_files types_and_params.h

add_files tb_data/data.txt
add_files tb_data/targets.txt

add_files neuro_hls_functions/bit_type.h
add_files neuro_hls_functions/dense.h

add_files -tb testbench.cpp

# Cria solução (padrão)
open_solution "sol1"

# (Opcional) clock — não é necessário para C-simulação
# create_clock -period 10

# Executa C Simulation
csim_design

# Fecha
exit