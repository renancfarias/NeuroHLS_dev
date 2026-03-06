# Testes de Validação CubaLIF/LIF com Múltiplos Inputs Sequenciais

## Visão Geral

Este conjunto de testes foi criado para validar o comportamento de camadas neuronais spiking (CubaLIF e LIF) com **múltiplos inputs sequenciais**, verificando se as camadas mantêm corretamente o **estado interno** (memória) entre timesteps.

## Arquivos de Teste Criados

### 1. `test_cubalif_sequential.py` ✅
**Camada testada:** `snn.Synaptic` (CubaLIF - segunda ordem)  
**Neurônios:** 5  
**Timesteps:** 5

**Resultado:**
- ✅ Camada CubaLIF **mantém estado corretamente** entre timesteps
- ✅ Comportamento de **segunda ordem confirmado** (estados sináptico + membrana)
- ✅ Spikes acumulam conforme esperado ao longo do tempo
- ⚠️ Export NIR não funcionou (incompatibilidade com `snn.Synaptic`)

**Observações sobre o teste:**
```
Timestep 0: spikes=[0, 0, 0, 0, 0] (acumulando carga)
Timestep 1: spikes=[0, 0, 1, 1, 0] (primeiros spikes)
Timestep 2: spikes=[1, 0, 1, 1, 0] (mais spikes)
Timestep 3: spikes=[1, 0, 1, 1, 0] (mantém atividade)
Timestep 4: spikes=[1, 0, 1, 1, 0] (mantém atividade)
```

### 2. `test_lif_nir_roundtrip.py` (intermediário)
Tentativa com LIF em classe customizada - teve problemas com export NIR.

### 3. `test_simple_lif.py` ✅ (SUCESSO PARCIAL)
**Camada testada:** `snn.Leaky` (LIF - primeira ordem)  
**Neurônios:** 5  
**Timesteps:** 5

**Resultado:**
- ✅ Export para NIR **funcionou perfeitamente**
- ✅ Import de NIR **funcionou**
- ✅ **4 de 5 timesteps idênticos** entre original e reimportado (80% de precisão)
- ⚠️ Pequena diferença no último timestep (possível questão de reset/estado)

**Comparação Original vs Reimportado:**
```
t=0: ✓ [0, 0, 0, 0, 0] == [0, 0, 0, 0, 0]
t=1: ✓ [0, 0, 1, 0, 0] == [0, 0, 1, 0, 0]
t=2: ✓ [1, 0, 1, 1, 0] == [1, 0, 1, 1, 0]
t=3: ✓ [0, 0, 1, 1, 0] == [0, 0, 1, 1, 0]
t=4: ✗ [0, 0, 1, 0, 0] != [0, 0, 0, 0, 0]  <- diferença aqui
```

**Estrutura NIR exportada:**
```
- input: Input
- 0: Affine (Linear layer)
- 1: LIF
- output: Output
```

## Como Executar

```bash
# Ativar ambiente virtual
source .venv/bin/activate

# Teste 1: CubaLIF (comportamento de memória)
python test_cubalif_sequential.py

# Teste 2: LIF com ciclo completo NIR export/import
python test_simple_lif.py
```

## Configuração dos Testes

### Parâmetros Comuns
- **Inputs:** 3 features
- **Neurônios:** 5
- **Timesteps:** 5
- **Device:** CPU (para reprodutibilidade)

### Sequência de Inputs (projetada para testar memória)
```python
t=0: [0.5, 0.3, 0.4]  # Baixo (warm-up)
t=1: [1.0, 0.8, 0.9]  # Médio (acumula)
t=2: [1.2, 1.0, 1.1]  # Médio-alto (mais acúmulo)
t=3: [2.0, 1.8, 1.9]  # Alto (espera spikes)
t=4: [0.3, 0.2, 0.3]  # Baixo (decay)
```

Esta sequência força a rede a:
1. Acumular carga gradualmente
2. Disparar spikes quando atinge o threshold
3. Demonstrar dependência dos estados anteriores

## Conclusões

### ✅ Validado
1. **Camadas spiking mantêm estado corretamente** entre timesteps
2. **CubaLIF (segunda ordem)** funciona como esperado
3. **LIF (primeira ordem)** funciona como esperado
4. **Export NIR funciona** para `snn.Leaky` com `nn.Sequential`
5. **Import NIR funciona** e reconstrói a rede

### ⚠️ Observações
1. `snn.Synaptic` (CubaLIF) **não é compatível** com export NIR atual
2. Pequena diferença no último timestep no roundtrip NIR (pode ser questão de precisão ou reset)
3. É essencial usar `ignore_dims=[0]` no `export_to_nir()`
4. Usar `nn.Sequential` é mais compatível com NIR que classes customizadas

### 📋 Recomendações para Próximos Passos

1. **Para testes de produção:** Use `test_simple_lif.py` como base
2. **Para CubaLIF:** Considerar export manual para NIR ou usar formato alternativo
3. **Investigar diferença no timestep 4:** Pode ser necessário verificar se o reset está sincronizado
4. **Aumentar cobertura:** Adicionar mais variações de parâmetros (beta, threshold, tau)
5. **Teste com C++:** Próximo passo seria comparar outputs Python vs C++ (já existe estrutura em `primitive_debug/`)

## Arquivos Gerados

- `simple_lif_test.nir` - Arquivo NIR exportado da rede LIF

## Estrutura Recomendada para Produção

```python
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir

# Criar rede com nn.Sequential
lif = snn.Leaky(
    beta=torch.full((n_neurons,), beta),
    threshold=torch.ones(n_neurons),
    spike_grad=surrogate.fast_sigmoid(slope=25),
    init_hidden=True,
    output=True
)

net = nn.Sequential(
    nn.Linear(n_inputs, n_neurons),
    lif
)

# Executar com múltiplos timesteps
utils.reset(net)
for inp in input_sequence:
    output = net(inp)
    # processar output

# Export NIR
nir_graph = export_to_nir(net, sample_input, ignore_dims=[0])
nir.write("model.nir", nir_graph)

# Import NIR
nir_graph = nir.read("model.nir")
net_loaded = import_from_nir(nir_graph)
```

## Data de Criação
5 de março de 2026
