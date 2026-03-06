"""
Teste LIF com Export/Import NIR - Versão Simplificada
------------------------------------------------------
Teste simples com LIF (não CubaLIF) para validar o ciclo completo:
1. Criar rede snnTorch (Linear + LIF)
2. Executar com múltiplos inputs
3. Exportar para NIR
4. Reimportar de NIR
5. Comparar outputs

Como LIF é de primeira ordem (apenas membrana), é mais simples
mas ainda testa o comportamento de memória entre timesteps.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import nir
from snntorch.export_nir import export_to_nir
from snntorch import import_nir
from pathlib import Path

print("=" * 80)
print("TESTE LIF - EXPORT/IMPORT NIR COM MÚLTIPLOS TIMESTEPS")
print("=" * 80)

# =========================
# CONFIGURAÇÃO
# =========================
n_inputs = 3
n_neurons = 5
num_timesteps = 5
device = torch.device("cpu")
beta = 0.9  # decay da membrana
threshold = 1.0

print(f"\nConfiguração:")
print(f"  Inputs: {n_inputs}")
print(f"  Neurônios: {n_neurons}")
print(f"  Timesteps: {num_timesteps}")
print(f"  Beta (decay): {beta}")
print(f"  Threshold: {threshold}")

# =========================
# CRIAR REDE ORIGINAL
# =========================
print("\n" + "=" * 80)
print("PASSO 1: Criando rede snnTorch original")
print("=" * 80)

class SimpleLIFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(n_inputs, n_neurons, bias=True)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=True,
            reset_mechanism='subtract'
        )
    
    def forward(self, x):
        # Garantir dimensão correta
        if x.dim() == 1:
            x = x.unsqueeze(0)
        cur = self.fc(x)
        spk = self.lif(cur)
        mem = self.lif.mem
        return spk, mem

net_original = SimpleLIFNet().to(device)

# Pesos fixos para reprodutibilidade
torch.manual_seed(42)
with torch.no_grad():
    net_original.fc.weight.data = torch.randn(n_neurons, n_inputs) * 0.5
    net_original.fc.bias.data = torch.randn(n_neurons) * 0.1

print(f"✓ Rede criada: Linear({n_inputs}→{n_neurons}) + LIF")

# =========================
# GERAR INPUTS DE TESTE
# =========================
print("\n" + "=" * 80)
print("PASSO 2: Gerando inputs de teste")
print("=" * 80)

torch.manual_seed(42)
input_sequence = [
    torch.tensor([[0.5, 0.3, 0.4]], dtype=torch.float32, device=device),
    torch.tensor([[1.0, 0.8, 0.9]], dtype=torch.float32, device=device),
    torch.tensor([[1.2, 1.0, 1.1]], dtype=torch.float32, device=device),
    torch.tensor([[2.0, 1.8, 1.9]], dtype=torch.float32, device=device),
    torch.tensor([[0.3, 0.2, 0.3]], dtype=torch.float32, device=device),
]

print("Inputs gerados:")
for t, inp in enumerate(input_sequence):
    print(f"  t={t}: {inp.squeeze().tolist()}")

# =========================
# EXECUTAR REDE ORIGINAL
# =========================
print("\n" + "=" * 80)
print("PASSO 3: Executando rede original")
print("=" * 80)

net_original.lif.mem = torch.zeros(1, n_neurons, device=device)

results_original = []

for t, inp in enumerate(input_sequence):
    spk, mem = net_original(inp)
    
    print(f"\n  t={t}: input={inp.squeeze().tolist()}")
    print(f"       spikes={spk.squeeze().detach().cpu().numpy().astype(int).tolist()}")
    print(f"       mem={mem.squeeze().detach().cpu().numpy()}")
    
    results_original.append({
        'timestep': t,
        'spikes': spk.squeeze().detach().cpu().numpy(),
        'mem': mem.squeeze().detach().cpu().numpy()
    })

print("\n✓ Execução original completa")

# =========================
# EXPORTAR PARA NIR
# =========================
print("\n" + "=" * 80)
print("PASSO 4: Exportando para NIR")
print("=" * 80)

try:
    # Usar input COM dimensão de batch [1, 3]
    sample_input = input_sequence[0]  # Mantém [1, 3]
    
    # Reset antes de exportar
    net_original.lif.mem = torch.zeros(1, n_neurons, device=device)
    
    # IMPORTANTE: ignore_dims=[0] é necessário para ignorar a dimensão de batch
    nir_graph = export_to_nir(net_original, sample_input, model_name="lif_test", ignore_dims=[0])
    nir_path = Path(__file__).parent / "lif_nir_test.nir"
    nir.write(str(nir_path), nir_graph)
    
    print(f"✓ NIR exportado: {nir_path.name}")
    print(f"\nNós do grafo NIR:")
    for node_name, node in nir_graph.nodes.items():
        print(f"  - {node_name}: {type(node).__name__}")
    
except Exception as e:
    print(f"✗ Erro ao exportar: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# =========================
# REIMPORTAR DE NIR
# =========================
print("\n" + "=" * 80)
print("PASSO 5: Reimportando de NIR")
print("=" * 80)

try:
    nir_graph_loaded = nir.read(str(nir_path))
    net_reimported = import_nir(nir_graph_loaded, torch.device("cpu"))
    net_reimported.eval()
    
    print(f"✓ Rede reimportada")
    
    # =========================
    # EXECUTAR REDE REIMPORTADA
    # =========================
    print("\n" + "=" * 80)
    print("PASSO 6: Executando rede reimportada")
    print("=" * 80)
    
    results_reimported = []
    
    for t, inp in enumerate(input_sequence):
        # Executar rede reimportada
        with torch.no_grad():
            output = net_reimported(inp)
        
        # Output pode ser tuple ou tensor
        if isinstance(output, tuple):
            spk_reimp = output[0]
        else:
            spk_reimp = output
        
        print(f"\n  t={t}: input={inp.squeeze().tolist()}")
        print(f"       spikes={spk_reimp.squeeze().cpu().numpy().astype(int).tolist()}")
        
        results_reimported.append({
            'timestep': t,
            'spikes': spk_reimp.squeeze().detach().cpu().numpy()
        })
    
    print("\n✓ Execução reimportada completa")
    
    # =========================
    # COMPARAÇÃO
    # =========================
    print("\n" + "=" * 80)
    print("PASSO 7: COMPARAÇÃO DE RESULTADOS")
    print("=" * 80)
    
    all_match = True
    
    for t in range(num_timesteps):
        orig = results_original[t]
        reimp = results_reimported[t]
        
        spikes_match = np.array_equal(
            orig['spikes'].astype(int),
            reimp['spikes'].astype(int)
        )
        
        match_symbol = "✓" if spikes_match else "✗"
        
        print(f"\nTimestep {t}: {match_symbol}")
        print(f"  Original:    {orig['spikes'].astype(int).tolist()}")
        print(f"  Reimportado: {reimp['spikes'].astype(int).tolist()}")
        
        if not spikes_match:
            all_match = False
            print(f"  ⚠ DIFERENÇA!")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✓✓✓ SUCESSO TOTAL ✓✓✓")
        print("Todos os outputs são idênticos!")
        print("O ciclo NIR export → import está funcionando perfeitamente")
        print("e a rede mantém estado entre timesteps corretamente.")
    else:
        print("✗ FALHA: Diferenças encontradas")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ Erro: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TESTE CONCLUÍDO")
print("=" * 80)
