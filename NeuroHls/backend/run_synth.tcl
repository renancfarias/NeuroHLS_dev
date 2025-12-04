# ============================================
# Script TCL para rodar SÍNTESE no Vitis HLS
# ============================================

# Cria projeto
open_project -reset projeto_teste_26_nov

# Define a função top (essa é a função que será sintetizada no futuro)
set_top snn_to_hls

# Adiciona os arquivos fonte do módulo

add_files snn_implementation.cpp
add_files snn_implementation.h
add_files weights.h
add_files types_and_params.h

add_files tb_data/data.txt
add_files tb_data/targets.txt

add_files neuro_hls_functions/bit_type.h
add_files neuro_hls_functions/dense.h

add_files -tb testbench.cpp

# Cria solução
open_solution "sol1"

# Configura clock do hardware (IMPORTANTE)
create_clock -period 10 -name default

# (Opcional) Define FPGA alvo (board/part)
set_part {xc7z020clg400-1}

# Executa SÍNTESE propriamente dita
csynth_design

# Sai da ferramenta
exit