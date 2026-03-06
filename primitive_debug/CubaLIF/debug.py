import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

class CubaLIF_Debug(nn.Module):
    """
    Implementação Python do CubaLIF para debug
    Segue EXATAMENTE a lógica do dense.h
    """
    def __init__(self, n_neurons, tau_syn, tau_mem, R, v_leak, v_threshold, v_reset, w_in, dt):
        super().__init__()
        
        # Parâmetros do modelo
        self.tau_syn = torch.tensor(tau_syn, dtype=torch.float32)
        self.tau_mem = torch.tensor(tau_mem, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)
        self.v_leak = torch.tensor(v_leak, dtype=torch.float32)
        self.v_threshold = torch.tensor(v_threshold, dtype=torch.float32)
        self.v_reset = torch.tensor(v_reset, dtype=torch.float32)
        self.w_in = torch.tensor(w_in, dtype=torch.float32)
        self.dt = torch.tensor(dt, dtype=torch.float32)
        
        # Estados (memória)
        self.u_state = torch.zeros(n_neurons, dtype=torch.float32)
        self.v_state = torch.zeros(n_neurons, dtype=torch.float32)
        
    def forward(self, input_current):
        """
        Aplica CubaLIF EXATAMENTE como no dense.h
        """
        spikes = torch.zeros_like(self.u_state, dtype=torch.bool)
        
        for i in range(len(input_current)):
            # ESTÁGIO 1: Dinâmica da Sinapse (u)
            leak_u = 0 - self.u_state[i]
            input_u = self.w_in[i] * input_current[i]
            du = (self.dt / self.tau_syn[i]) * (leak_u + input_u)
            self.u_state[i] = self.u_state[i] + du
            
            # ESTÁGIO 2: Dinâmica da Membrana (v)
            leak_v = self.v_leak[i] - self.v_state[i]
            input_v = self.R[i] * self.u_state[i]
            dv = (self.dt / self.tau_mem[i]) * (leak_v + input_v)
            self.v_state[i] = self.v_state[i] + dv
            
            # ESTÁGIO 3: Disparo e Reset
            if self.v_state[i] >= self.v_threshold[i]:
                spikes[i] = True
                self.v_state[i] = self.v_reset[i]
                # Nota: u_state NÃO é resetado
            else:
                spikes[i] = False
        
        return spikes
    
    def reset_states(self):
        """Reset dos estados para novo teste"""
        self.u_state.zero_()
        self.v_state.zero_()


print("=" * 70)
print("Teste CubaLIF - Comparação Python vs C++")
print("=" * 70)

# Configuração do teste: 3 neurônios
n_neurons = 3

# Parâmetros típicos de um neurônio CubaLIF
tau_syn = [5.0, 5.0, 5.0]      # ms
tau_mem = [10.0, 10.0, 10.0]   # ms
R_mem = [1.0, 1.0, 1.0]        # Resistência
v_leak = [0.0, 0.0, 0.0]       # Voltagem de repouso
v_threshold = [1.0, 1.0, 1.0]  # Limiar de disparo
v_reset = [0.0, 0.0, 0.0]      # Voltagem após spike
w_in = [0.5, 1.0, 1.5]         # Pesos sinápticos (diferentes para cada neurônio)
dt = 1.0                        # Passo de tempo (ms)

# Criar modelo
model = CubaLIF_Debug(n_neurons, tau_syn, tau_mem, R_mem, v_leak, v_threshold, v_reset, w_in, dt)

print(f"\nConfiguração:")
print(f"  Neurônios: {n_neurons}")
print(f"  tau_syn: {tau_syn}")
print(f"  tau_mem: {tau_mem}")
print(f"  R: {R_mem}")
print(f"  v_leak: {v_leak}")
print(f"  v_threshold: {v_threshold}")
print(f"  v_reset: {v_reset}")
print(f"  w_in: {w_in}")
print(f"  dt: {dt}")

# Sequência de testes ao longo do tempo
print("\n" + "=" * 70)
print("Simulação Temporal (10 timesteps)")
print("=" * 70)

# Corrente de entrada constante
input_sequence = [
    torch.tensor([2.0, 2.0, 2.0]),  # t=0
    torch.tensor([2.0, 2.0, 2.0]),  # t=1
    torch.tensor([2.0, 2.0, 2.0]),  # t=2
    torch.tensor([2.0, 2.0, 2.0]),  # t=3
    torch.tensor([2.0, 2.0, 2.0]),  # t=4
    torch.tensor([0.0, 0.0, 0.0]),  # t=5 (input zero)
    torch.tensor([0.0, 0.0, 0.0]),  # t=6
    torch.tensor([3.0, 3.0, 3.0]),  # t=7 (input maior)
    torch.tensor([3.0, 3.0, 3.0]),  # t=8
    torch.tensor([0.0, 0.0, 0.0]),  # t=9
]

results = []

for t, input_current in enumerate(input_sequence):
    spikes = model.forward(input_current)
    
    print(f"\nTimestep {t}:")
    print(f"  Input: {input_current.numpy()}")
    print(f"  u_state: {model.u_state.detach().numpy()}")
    print(f"  v_state: {model.v_state.detach().numpy()}")
    print(f"  Spikes: {spikes.detach().numpy().astype(int)}")
    
    results.append({
        'timestep': t,
        'input': input_current.numpy(),
        'u_state': model.u_state.detach().numpy().copy(),
        'v_state': model.v_state.detach().numpy().copy(),
        'spikes': spikes.detach().numpy().astype(int)
    })

# Salvar resultados para comparação com C++
print("\n" + "=" * 70)
print("Salvando resultados...")
print("=" * 70)

with open("primitive_debug/CubaLIF/out_py.txt", "w") as f:
    f.write("CubaLIF Test Results\n")
    f.write("=" * 70 + "\n\n")
    
    for r in results:
        f.write(f"Timestep {r['timestep']}:\n")
        f.write(f"  Input: [{', '.join([f'{x:.6f}' for x in r['input']])}]\n")
        f.write(f"  u_state: [{', '.join([f'{x:.6f}' for x in r['u_state']])}]\n")
        f.write(f"  v_state: [{', '.join([f'{x:.6f}' for x in r['v_state']])}]\n")
        f.write(f"  Spikes: [{', '.join([str(x) for x in r['spikes']])}]\n")
        f.write("\n")

print("\nResultados salvos em: primitive_debug/CubaLIF/out_py.txt")

# Teste 2: Caso crítico - Verificar se spike reseta apenas v, não u
print("\n" + "=" * 70)
print("Teste 2: Verificação de Reset (v_state reseta, u_state não)")
print("=" * 70)

model.reset_states()

# Input grande para forçar spike rápido
test_input = torch.tensor([5.0, 5.0, 5.0])

print(f"\nInput forte: {test_input.numpy()}")
for t in range(3):
    spikes = model.forward(test_input)
    print(f"  t={t}: u={model.u_state.numpy()}, v={model.v_state.numpy()}, spike={spikes.numpy().astype(int)}")
    
print("\nObservação: Após spike, v_state vai para v_reset, mas u_state continua acumulando!")

# Executa implementação C++
from primitive_debug import run_cpp_impl
run_cpp_impl()
