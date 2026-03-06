"""
Teste Simples - LIF com nn.Sequential
--------------------------------------
Versão minimalista seguindo o padrão exato do código que funciona no projeto.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
import numpy as np
import nir
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir
from pathlib import Path

print("=" * 80)
print("TESTE LIF SIMPLES - nn.Sequential")
print("=" * 80)

device = torch.device("cpu")
beta = 0.9
threshold = 1.0
spike_grad = surrogate.fast_sigmoid(slope=25)

# =========================
# CRIAR REDE USANDO nn.Sequential
# =========================
print("\nCriando rede usando nn.Sequential...")

# LIF para 5 neurônios (shape=(5,))
lif1 = snn.Leaky(
    beta=torch.full((5,), beta, device=device),
    threshold=torch.ones(5, device=device),
    spike_grad=spike_grad,
    init_hidden=True,
    output=True
)

net = nn.Sequential(
    nn.Linear(3, 5, bias=True),
    lif1
).to(device)

# Pesos fixos
torch.manual_seed(42)
with torch.no_grad():
    net[0].weight.data = torch.randn(5, 3) * 0.5
    net[0].bias.data = torch.randn(5) * 0.1

print("✓ Rede criada: Linear(3→5) + LIF(5)")

# =========================
# INPUTS DE TESTE
# =========================
torch.manual_seed(42)
inputs = [
    torch.tensor([[0.5, 0.3, 0.4]], device=device),
    torch.tensor([[1.0, 0.8, 0.9]], device=device),
    torch.tensor([[1.2, 1.0, 1.1]], device=device),
    torch.tensor([[2.0, 1.8, 1.9]], device=device),
    torch.tensor([[0.3, 0.2, 0.3]], device=device),
]

# =========================
# EXECUÇÃO ORIGINAL
# =========================
print("\n" + "=" * 80)
print("EXECUÇÃO ORIGINAL")
print("=" * 80)

utils.reset(net)
results_orig = []

for t, inp in enumerate(inputs):
    out = net(inp)
    # net pode retornar (spk, mem) se init_hidden=True
    if isinstance(out, tuple):
        spk = out[0]
    else:
        spk = out
    
    print(f"\nt={t}: input={inp.squeeze().tolist()}")
    print(f"     spikes={spk.squeeze().detach().cpu().numpy().astype(int).tolist()}")
    print(f"     mem={lif1.mem.squeeze().detach().cpu().numpy()}")
    results_orig.append(spk.squeeze().detach().cpu().numpy().copy())

# =========================
# EXPORTAR PARA NIR
# =========================
print("\n" + "=" * 80)
print("EXPORTANDO PARA NIR")
print("=" * 80)

net.eval()
sample = torch.randn(1, 3)

try:
    nir_graph = export_to_nir(net, sample, ignore_dims=[0])
    nir_path = Path(__file__).parent / "simple_lif_test.nir"
    nir.write(str(nir_path), nir_graph)
    
    print(f"✓ NIR exportado: {nir_path.name}")
    print("\nNós NIR:")
    for name, node in nir_graph.nodes.items():
        print(f"  - {name}: {type(node).__name__}")
    
except Exception as e:
    print(f"✗ Erro ao exportar: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# =========================
# REIMPORTAR DE NIR
# =========================
print("\n" + "=" * 80)
print("REIMPORTANDO DE NIR")
print("=" * 80)

try:
    nir_graph_loaded = nir.read(str(nir_path))
    net_reimported = import_from_nir(nir_graph_loaded)
    net_reimported = net_reimported.to(device)
    net_reimported.eval()
    
    print("✓ Rede reimportada")
    
except Exception as e:
    print(f"✗ Erro ao reimportar: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# =========================
# EXECUTAR REDE REIMPORTADA
# =========================
print("\n" + "=" * 80)
print("EXECUÇÃO REIMPORTADA")
print("=" * 80)

# Resetar estado da rede reimportada
utils.reset(net_reimported)

results_reimp = []

with torch.no_grad():
    for t, inp in enumerate(inputs):
        spk = net_reimported(inp)
        if isinstance(spk, tuple):
            spk = spk[0]
        
        print(f"\nt={t}: input={inp.squeeze().tolist()}")
        print(f"     spikes={spk.squeeze().cpu().numpy().astype(int).tolist()}")
        results_reimp.append(spk.squeeze().cpu().numpy().copy())

# =========================
# COMPARAÇÃO
# =========================
print("\n" + "=" * 80)
print("COMPARAÇÃO")
print("=" * 80)

all_match = True
for t in range(len(inputs)):
    orig = results_orig[t].astype(int)
    reimp = results_reimp[t].astype(int)
    
    match = np.array_equal(orig, reimp)
    symbol = "✓" if match else "✗"
    
    print(f"\nt={t}: {symbol}")
    print(f"  Original:    {orig.tolist()}")
    print(f"  Reimportado: {reimp.tolist()}")
    
    if not match:
        all_match = False

print("\n" + "=" * 80)
if all_match:
    print("✓✓✓ SUCESSO ✓✓✓")
    print("Todos os outputs são idênticos!")
else:
    print("✗ Diferenças encontradas")
print("=" * 80)
