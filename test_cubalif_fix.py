"""
Script para testar as correções no carregamento e geração de código CubaLIF
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from neuro_hls import NeuroHls

print("=" * 70)
print("TESTE DE CORREÇÕES - CubaLIF")
print("=" * 70)

# Criar pasta de teste
test_folder = "z_test_cubalifefix"

# Inicializar NeuroHLS
neuro_hls = NeuroHls(test_folder)

# Carregar o arquivo .nir
print("\n[1] Carregando arquivo .nir...")
model = neuro_hls.read_nir_file("braille_noDelay_bias_zero.nir")

print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total de camadas: {len(model.layers)}")

# Gerar implementação
print("\n[2] Gerando implementação C++...")
neuro_hls.implement_model(model)

print("\n[3] Verificando arquivo gerado...")

# Ler o arquivo gerado e procurar por CubaLIF
cpp_file = Path(test_folder) / "snn_implementation.cpp"
if cpp_file.exists():
    with open(cpp_file, 'r') as f:
        content = f.read()
    
    print("\n--- Chamadas CubaLIF encontradas ---")
    for i, line in enumerate(content.split('\n'), 1):
        if 'CubaLIF' in line:
            print(f"Linha {i}: {line.strip()}")
    
    print("\n--- Declarações de estados (u_state, v_state) ---")
    for i, line in enumerate(content.split('\n'), 1):
        if 'u_state' in line or 'v_state' in line:
            print(f"Linha {i}: {line.strip()}")
    
    print("\n[4] Verificando neuron_params.h...")
    params_file = Path(test_folder) / "neuron_params.h"
    if params_file.exists():
        with open(params_file, 'r') as f:
            params_content = f.read()
        
        print("\n--- Parâmetros tau_syn, tau_mem, dt ---")
        for i, line in enumerate(params_content.split('\n'), 1):
            if 'tau_syn' in line or 'tau_mem' in line or 'dt_' in line or 'R_mem' in line:
                print(f"Linha {i}: {line.strip()}")
    
    print("\n" + "=" * 70)
    print("TESTE CONCLUÍDO")
    print("=" * 70)
    print(f"\nArquivos gerados em: {test_folder}/")
    print("  - snn_implementation.cpp")
    print("  - snn_implementation.h")
    print("  - neuron_params.h")
    print("  - quantization.h")
    
else:
    print(f"ERRO: Arquivo {cpp_file} não foi gerado!")
