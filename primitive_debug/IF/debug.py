import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from primitive_debug import run_cpp_impl

class IF_Debug(nn.Module):
    """
    Implementação Python do IF (Integrate-and-Fire) para debug
    Segue EXATAMENTE a lógica do dense.h
    """
    def __init__(self, n_neurons, R, v_threshold, v_reset):
        super().__init__()
        
        # Parâmetros do modelo
        self.R = torch.tensor(R, dtype=torch.float32)
        self.v_threshold = torch.tensor(v_threshold, dtype=torch.float32)
        self.v_reset = torch.tensor(v_reset, dtype=torch.float32)
        
        # Estado (memória do potencial de membrana)
        self.membrane_potential = torch.zeros(n_neurons, dtype=torch.float32)
        
    def forward(self, input_current):
        """
        Aplica IF EXATAMENTE como no dense.h
        
        Integrate-and-Fire:
        1. Integração: v = v + input * R
        2. Disparo: if v >= threshold então spike=1, v=v_reset
        """
        spikes = torch.zeros_like(self.membrane_potential, dtype=torch.bool)
        
        for i in range(len(input_current)):
            # 1. Integração
            self.membrane_potential[i] = self.membrane_potential[i] + (input_current[i] * self.R[i])
            
            # 2. Disparo e Reset
            if self.membrane_potential[i] >= self.v_threshold[i]:
                spikes[i] = True
                self.membrane_potential[i] = self.v_reset[i]
            else:
                spikes[i] = False
        
        return spikes
    
    def reset_states(self):
        """Reset dos estados para novo teste"""
        self.membrane_potential.zero_()


print("=" * 70)
print("Teste IF (Integrate-and-Fire) - Comparação Python vs C++")
print("=" * 70)

# Configuração do teste: 4 neurônios
n_neurons = 4

# Parâmetros hardcoded - mesmos do C++
R = [1.0, 1.0, 1.0, 1.0]
v_threshold = [4.0, 4.0, 4.0, 4.0]
v_reset = [0.0, 0.0, 0.0, 0.0]

# Criar modelo
model = IF_Debug(n_neurons, R, v_threshold, v_reset)

print(f"\nConfiguração:")
print(f"  Neurônios: {n_neurons}")
print(f"  R: {R}")
print(f"  v_threshold: {v_threshold}")
print(f"  v_reset: {v_reset}")

# Testes com 3 samples
print("\n" + "=" * 70)
print("Testes com 3 Samples")
print("=" * 70)

# Correntes de entrada
input_samples = [
    torch.tensor([1.5, 2.0, 1.0, 2.5]),   # Sample 1
    torch.tensor([2.0, 2.5, 3.0, 1.5]),   # Sample 2
    torch.tensor([1.0, 1.5, 2.0, 1.0]),   # Sample 3
]

results = []

for i, input_current in enumerate(input_samples, 1):
    spikes = model.forward(input_current)
    
    print(f"\nSample {i}:")
    print(f"  Input: {input_current.numpy()}")
    print(f"  Membrane: {model.membrane_potential.detach().numpy()}")
    print(f"  Spikes: {spikes.detach().numpy().astype(int)}")
    
    results.append({
        'sample': i,
        'input': input_current.numpy(),
        'membrane': model.membrane_potential.detach().numpy().copy(),
        'spikes': spikes.detach().numpy().astype(int)
    })

# Salvar resultados para comparação com C++
print("\n" + "=" * 70)
print("Salvando resultados...")
print("=" * 70)

with open("primitive_debug/IF/out_py.txt", "w") as f:
    f.write("*** IF Neuron Python Implementation ***\n")
    f.write("Parameters:\n")
    f.write(f"  R: [{R[0]}, {R[1]}, {R[2]}, {R[3]}]\n")
    f.write(f"  Threshold: [{v_threshold[0]}, {v_threshold[1]}, {v_threshold[2]}, {v_threshold[3]}]\n")
    f.write(f"  V_reset: {v_reset[0]}\n")
    f.write("\n")
    
    for r in results:
        f.write(f"Sample {r['sample']}\n")
        f.write(f"  Input: [{r['input'][0]}, {r['input'][1]}, {r['input'][2]}, {r['input'][3]}]\n")
        f.write(f"  Membrane: [{r['membrane'][0]}, {r['membrane'][1]}, {r['membrane'][2]}, {r['membrane'][3]}]\n")
        f.write(f"  Spikes: [{r['spikes'][0]}, {r['spikes'][1]}, {r['spikes'][2]}, {r['spikes'][3]}]\n")
        f.write("\n")

print("\nResultados salvos em: primitive_debug/IF/out_py.txt")

# Executa implementação C++
run_cpp_impl()