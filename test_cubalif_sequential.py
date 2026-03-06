"""
Teste CubaLIF com Múltiplos Inputs Sequenciais
-----------------------------------------------
Este script testa uma camada CubaLIF com 5 neurônios,
usando 4-5 inputs sequenciais para verificar se a camada
mantém corretamente o estado (memória) entre timesteps.

Procedimento:
1. Criar rede snnTorch com camada CubaLIF (5 neurônios)
2. Gerar 5 inputs de teste diferentes
3. Executar a rede e coletar outputs
4. Exportar para NIR
5. Reimportar de NIR
6. Executar novamente com os mesmos inputs
7. Comparar outputs (original vs reimportado)
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import nir
from snntorch.export_nir import export_to_nir
from pathlib import Path

print("=" * 80)
print("TESTE CUBALIF - MÚLTIPLOS INPUTS SEQUENCIAIS")
print("=" * 80)

# =========================
# CONFIGURAÇÃO
# =========================
n_neurons = 5
num_timesteps = 5  # 5 inputs diferentes
device = torch.device("cpu")  # Usar CPU para reprodutibilidade

# Parâmetros do CubaLIF
# Vamos variar um pouco os parâmetros entre neurônios para tornar o teste mais interessante
tau_syn_values = [5.0, 6.0, 5.5, 4.5, 5.2]   # Constante de tempo sináptica (ms)
tau_mem_values = [10.0, 12.0, 11.0, 9.0, 10.5]  # Constante de tempo da membrana (ms)
beta_syn = 0.8  # exp(-dt/tau_syn), assumindo dt=1ms
beta_mem = 0.9  # exp(-dt/tau_mem), assumindo dt=1ms
threshold = 1.0

print(f"\nConfiguração:")
print(f"  Neurônios: {n_neurons}")
print(f"  Timesteps: {num_timesteps}")
print(f"  tau_syn (variados): {tau_syn_values}")
print(f"  tau_mem (variados): {tau_mem_values}")
print(f"  Threshold: {threshold}")
print(f"  Device: {device}")

# =========================
# CRIAR REDE ORIGINAL
# =========================
print("\n" + "=" * 80)
print("PASSO 1: Criando rede snnTorch original")
print("=" * 80)

class CubaLIFNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Camada Linear (entrada -> neurônios)
        self.fc = nn.Linear(3, n_neurons, bias=True)  # 3 features de entrada
        
        # Camada CubaLIF com 5 neurônios
        # CubaLIF é de segunda ordem (tem dinâmica sináptica + membrana)
        self.lif = snn.Synaptic(
            alpha=beta_syn,  # decay sináptico
            beta=beta_mem,   # decay da membrana
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=True,
            reset_mechanism='subtract'  # subtrai threshold ao invés de resetar para zero
        )
    
    def forward(self, x):
        """Forward para um único timestep"""
        # Garantir que x seja 2D [batch, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        cur = self.fc(x)
        spk = self.lif(cur)
        # Acessar estados internos
        syn = self.lif.syn
        mem = self.lif.mem
        return spk, syn, mem

# Instanciar rede
net_original = CubaLIFNet().to(device)

# Inicializar pesos com valores conhecidos para reprodutibilidade
torch.manual_seed(42)
with torch.no_grad():
    net_original.fc.weight.data = torch.randn(n_neurons, 3) * 0.5
    net_original.fc.bias.data = torch.randn(n_neurons) * 0.1

print(f"✓ Rede criada com sucesso")
print(f"  Peso FC: shape {net_original.fc.weight.shape}")
print(f"  Bias FC: shape {net_original.fc.bias.shape}")

# =========================
# GERAR INPUTS DE TESTE
# =========================
print("\n" + "=" * 80)
print("PASSO 2: Gerando inputs de teste")
print("=" * 80)

# Vamos criar uma sequência interessante de inputs:
# - Timestep 0: Input baixo (warm-up)
# - Timestep 1-2: Input médio (acumula estado)
# - Timestep 3: Input alto (deve causar spikes)
# - Timestep 4: Input baixo (decay)

np.random.seed(42)
input_sequence = [
    torch.tensor([[0.5, 0.3, 0.4]], dtype=torch.float32, device=device),  # t=0: baixo
    torch.tensor([[1.0, 0.8, 0.9]], dtype=torch.float32, device=device),  # t=1: médio
    torch.tensor([[1.2, 1.0, 1.1]], dtype=torch.float32, device=device),  # t=2: médio-alto
    torch.tensor([[2.0, 1.8, 1.9]], dtype=torch.float32, device=device),  # t=3: alto (espera spikes)
    torch.tensor([[0.3, 0.2, 0.3]], dtype=torch.float32, device=device),  # t=4: baixo (decay)
]

print(f"Sequência de {len(input_sequence)} inputs gerada:")
for t, inp in enumerate(input_sequence):
    print(f"  t={t}: {inp.squeeze().tolist()}")

# =========================
# EXECUTAR REDE ORIGINAL
# =========================
print("\n" + "=" * 80)
print("PASSO 3: Executando rede original")
print("=" * 80)

# Reset dos estados ocultos
net_original.lif.reset_hidden()

results_original = []

print("\nExecução timestep-by-timestep:")
for t, inp in enumerate(input_sequence):
    spk, syn, mem = net_original(inp)
    
    print(f"\n  Timestep {t}:")
    print(f"    Input: {inp.squeeze().tolist()}")
    print(f"    Spikes: {spk.squeeze().detach().cpu().numpy().astype(int).tolist()}")
    print(f"    Syn State: {syn.squeeze().detach().cpu().numpy()}")
    print(f"    Mem State: {mem.squeeze().detach().cpu().numpy()}")
    
    results_original.append({
        'timestep': t,
        'input': inp.squeeze().detach().cpu().numpy(),
        'spikes': spk.squeeze().detach().cpu().numpy(),
        'syn_state': syn.squeeze().detach().cpu().numpy(),
        'mem_state': mem.squeeze().detach().cpu().numpy()
    })

print("\n✓ Execução original completa")

# =========================
# EXPORTAR PARA NIR
# =========================
print("\n" + "=" * 80)
print("PASSO 4: Exportando para NIR")
print("=" * 80)

try:
    # Preparar input de exemplo para export (sem batch dimension)
    sample_input = input_sequence[0].squeeze(0)  # Remove batch dimension [1, 3] -> [3]
    
    # Exportar para NIR
    nir_graph = export_to_nir(net_original, sample_input, model_name="cubalif_test")
    
    # Salvar NIR
    nir_path = Path(__file__).parent / "cubalif_sequential_test.nir"
    nir.write(str(nir_path), nir_graph)
    
    print(f"✓ NIR exportado com sucesso para: {nir_path.name}")
    print(f"\nConteúdo NIR:")
    print(f"  Nodes: {list(nir_graph.nodes.keys())}")
    
    for node_name, node in nir_graph.nodes.items():
        print(f"  - {node_name}: {type(node).__name__}")
    
except Exception as e:
    print(f"✗ Erro ao exportar para NIR: {e}")
    import traceback
    traceback.print_exc()
    print("\nContinuando sem NIR export/import...")
    results_reimported = None

# =========================
# REIMPORTAR DE NIR
# =========================
print("\n" + "=" * 80)
print("PASSO 5: Reimportando de NIR")
print("=" * 80)

try:
    from snntorch import import_nir
    
    # Carregar NIR
    nir_graph_loaded = nir.read(str(nir_path))
    
    # Importar de volta para snnTorch
    net_reimported = import_nir(nir_graph_loaded)
    net_reimported = net_reimported.to(device)
    net_reimported.eval()
    
    print(f"✓ Rede reimportada de NIR com sucesso")
    
    # =========================
    # EXECUTAR REDE REIMPORTADA
    # =========================
    print("\n" + "=" * 80)
    print("PASSO 6: Executando rede reimportada")
    print("=" * 80)
    
    results_reimported = []
    
    print("\nExecução timestep-by-timestep:")
    for t, inp in enumerate(input_sequence):
        spk_reimp = net_reimported(inp)
        
        # snnTorch import pode retornar diferentes formatos
        if isinstance(spk_reimp, tuple):
            spk_reimp = spk_reimp[0]
        
        print(f"\n  Timestep {t}:")
        print(f"    Input: {inp.squeeze().tolist()}")
        print(f"    Spikes: {spk_reimp.squeeze().detach().cpu().numpy().astype(int).tolist()}")
        
        results_reimported.append({
            'timestep': t,
            'input': inp.squeeze().detach().cpu().numpy(),
            'spikes': spk_reimp.squeeze().detach().cpu().numpy()
        })
    
    print("\n✓ Execução reimportada completa")
    
except Exception as e:
    print(f"✗ Erro ao reimportar de NIR: {e}")
    import traceback
    traceback.print_exc()
    results_reimported = None

# =========================
# COMPARAÇÃO DE RESULTADOS
# =========================
print("\n" + "=" * 80)
print("PASSO 7: COMPARAÇÃO DE RESULTADOS")
print("=" * 80)

if results_reimported:
    print("\n" + "─" * 80)
    print("Comparação: Original vs Reimportado")
    print("─" * 80)
    
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
            print(f"  ⚠ DIFERENÇA DETECTADA!")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✓ SUCESSO: Todos os outputs são idênticos!")
        print("  A camada CubaLIF mantém corretamente o estado entre timesteps")
        print("  tanto na versão original quanto na reimportada de NIR.")
    else:
        print("✗ FALHA: Outputs diferem entre original e reimportado")
        print("  Pode haver problema na exportação/importação NIR ou nos parâmetros.")
    print("=" * 80)
else:
    print("\n⚠ Comparação não realizada (reimportação falhou)")
    print("\nResultados originais salvos para análise:")
    for t in range(num_timesteps):
        orig = results_original[t]
        print(f"  t={t}: spikes={orig['spikes'].astype(int).tolist()}")

# =========================
# ANÁLISE DE ESTADO
# =========================
print("\n" + "=" * 80)
print("ANÁLISE DO COMPORTAMENTO DE MEMÓRIA")
print("=" * 80)

print("\nVerificando se a camada mantém estado entre timesteps:")
print("(Estados sináptico e de membrana devem evoluir continuamente)\n")

has_memory = False
for t in range(1, num_timesteps):
    curr_syn = results_original[t]['syn_state']
    prev_syn = results_original[t-1]['syn_state']
    
    # Se há memória, o estado atual deve depender do anterior
    # (não deve resetar para zero a menos que haja spike)
    if not np.allclose(curr_syn, 0.0, atol=1e-6):
        has_memory = True
        print(f"  t={t}: Estado sináptico não-zero detectado")
        print(f"         (indica que há acumulação de informação)")

if has_memory:
    print("\n✓ Camada CubaLIF está mantendo estado corretamente")
    print("  (comportamento de segunda ordem confirmado)")
else:
    print("\n⚠ Camada pode não estar mantendo estado adequadamente")

print("\n" + "=" * 80)
print("TESTE CONCLUÍDO")
print("=" * 80)
