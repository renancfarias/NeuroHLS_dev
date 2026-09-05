# Especificação do NeuroHLS

## 1. Visão geral

O NeuroHLS é uma ferramenta para converter um grafo no padrão
[Neuromorphic Intermediate Representation (NIR)](https://neuroir.org/) em um
programa C++ compatível com High-Level Synthesis (HLS).

O fluxo principal deve:

1. ler um modelo NIR e metadados opcionais;
2. inferir a topologia, as dependências e as dimensões das camadas;
3. gerar uma implementação C++ estática da rede;
4. gerar um testbench a partir de um conjunto de dados;
5. executar simulação C, síntese e co-simulação no Vitis HLS;
6. disponibilizar estimativas de desempenho, recursos e potência.

Esta especificação descreve tanto o comportamento já disponível quanto as
funções futuras necessárias para completar o fluxo.

## 2. Objetivos

- Permitir que redes neurais pulsadas descritas em NIR sejam sintetizadas em
  FPGA sem reescrita manual da arquitetura.
- Preservar pesos, parâmetros neuronais, dimensões, conexões recorrentes e
  mecanismo de reset descritos pelo modelo.
- Gerar código com dimensões conhecidas em tempo de compilação, adequado ao
  Vitis HLS.
- Suportar execução densa, baseada em passos de tempo, e execução orientada a
  eventos.
- Permitir comparação entre o modelo de referência e a implementação C++.

## 3. Não objetivos

- Treinar ou otimizar os pesos da rede.
- Substituir o formato NIR por um formato proprietário.
- Garantir suporte imediato a toda primitiva presente no ecossistema NIR.
- Ocultar diferenças numéricas causadas por quantização; elas devem ser
  medidas e documentadas.
- Gerar hardware dinamicamente dimensionado em tempo de execução.

## 4. Terminologia

- **Frontend NIR**: leitura do grafo e criação da configuração interna.
- **Backend time-driven**: execução de todos os elementos a cada passo de tempo.
- **Backend orientado a eventos**: execução propagada por eventos de spike.
- **Self-event**: evento interno agendado analiticamente para um cruzamento de
  limiar futuro.
- **Active List**: lista compacta dos neurônios cujo estado ainda precisa ser
  atualizado em ticks discretos.
- **Lightweight tick**: atualização disparada pelo watermark `END_STEP`,
  aplicada apenas aos IDs presentes na Active List.
- **Primitiva**: implementação HLS de uma camada ou modelo neuronal.
- **Suporte de ponta a ponta**: leitura NIR, geração de C++, compilação e teste
  de equivalência funcional.
- **Estado neuronal**: potencial de membrana e, quando aplicável, corrente
  sináptica mantidos entre passos.

## 5. Requisitos funcionais

### RF-01 — Inicialização do projeto

`NeuroHls(folder_path, settings64_path=None)` deve criar o diretório de saída,
quando necessário, e copiar para ele os arquivos do backend HLS.

- Arquivos existentes do usuário não devem ser removidos silenciosamente.
- `settings64_path` deve ser opcional e usado para configurar ferramentas AMD
  no Windows.

### RF-02 — Leitura do modelo NIR

`read_nir_file(nir_file_path, metadata_file_path=None)` deve:

- carregar um grafo com `nir.read`;
- iniciar a travessia no nó chamado `input`;
- inferir shapes ausentes a partir de nós adjacentes;
- identificar dependências diretas e recorrentes;
- criar uma representação `ModelConfig` ordenada para geração de código;
- rejeitar tipos de nó desconhecidos com mensagem que identifique o tipo;
- rejeitar shapes que não possam ser inferidos.

O grafo deve possuir nós `input` e `output`. Cada entrada e saída utilizada
pelo gerador deve ter shape estático.

### RF-03 — Metadados externos

Metadados JSON devem poder ser informados explicitamente ou descobertos no
mesmo diretório do NIR.

A descoberta automática deve procurar, nesta ordem:

1. `<nome-do-modelo>.metadata.json`;
2. arquivos `*.metadata.json` que apontem para o NIR por `model_file`,
   `nir_file`, `file` ou `path`.

Metadados globais e específicos por camada podem ser definidos em `layers`,
`nodes` ou `node_metadata`. Metadados da camada têm precedência sobre os
metadados já presentes no nó NIR.

Para `CubaLIF`, os seguintes formatos devem selecionar reset por subtração:

- `reset_mechanism: "subtract"`;
- `reset_by_subtraction` com valor booleano verdadeiro;
- strings equivalentes a verdadeiro: `1`, `true`, `yes` ou `y`.

Na ausência desses valores, deve ser usado reset para `v_reset`.

### RF-04 — Configuração e quantização

`ModelConfig` deve armazenar as camadas, shapes de entrada e saída e a
quantização do modelo.

Devem existir configurações independentes para:

- entrada;
- pesos e parâmetros;
- potenciais internos.

Para a CUBA-LIF event-driven também devem existir configurações independentes
para a estratégia de evolução neuronal e para o limiar de desativação:

```python
model.define_event_cuba_lif_strategy("active_list")
model.define_event_active_noise_threshold(valor)
```

O único nome canônico de estratégia é `active_list`. O limiar
de ruído deve ser finito e maior que zero; o padrão é `1e-6`.

Cada configuração usa `(total_bits, integer_bits)`. O padrão é `(16, 8)`.
Quando `use_float=True`, o código gerado deve usar `float`. Em ponto fixo,
devem ser usados tipos `ap_fixed`.

Valores inválidos devem ser rejeitados: `total_bits > 0`, `integer_bits > 0`
e `integer_bits <= total_bits`.

### RF-04A — Paralelismo time-driven por percentagem e reúso

O parâmetro público `p` do backend time-driven deve ser um número finito em
`[0, 1]` que representa a fração solicitada do domínio estático de operações
de cada camada. Ele não representa somente o número de saídas nem é uma porta
em tempo de execução.

Para um domínio estático de `W` operações, o gerador deve resolver:

```text
U = 1                                             se p = 0
U = min(W, max(1, floor(p * W + 0.5)))            se 0 < p <= 1
R = ceil(W / U)
p_eff = U / W
I = U * R - W
```

`U` é o número de elementos de processamento, `R` é o número de grupos de
reúso, `p_eff` é a percentagem realmente representável e `I` são as posições
ociosas do último grupo. `p=0` é o sentinela explícito para uma unidade
serial. O domínio `W` deve contar MACs para `Linear`/`Affine` e convoluções,
acumulações de janela para pooling, e elementos ou atualizações de neurônios
para operadores elemento a elemento e neuronais.

As APIs canônicas são:

```python
model.define_time_driven_parallelism(0.025)
model.define_time_driven_layer_parallelism("linear_1", 0.01)
```

`define_layer_parallelism` é um alias obsoleto temporário. Não deve existir
uma chave independente de paralelismo de redução: o mesmo `p` cobre o domínio
inteiro das reduções. A geração time-driven deve registrar, em
`parallelism_manifest.json`, `W`, `U`, `R`, `p_eff`, `I`, o operador e o valor
solicitado para cada camada aplicável.

O `II` efetivamente atingido, a latência e a utilização de recursos devem ser
obtidos dos relatórios HLS; `R` é uma característica do agendamento estático,
não uma garantia de desempenho físico.

### RF-04B — Escopo do paralelismo event-driven

O backend event-driven não possui parâmetro configurável de paralelismo. Cada
stream transfere um registro por operação de interface; os atores continuam
concorrentes por `DATAFLOW`, FIFOs e pipelines internos escalares. Chamadas de
compatibilidade com `define_event_driven_parallelism(0)` podem emitir aviso de
obsolescência; valores não nulos devem ser rejeitados, em vez de serem
ignorados silenciosamente.

### RF-05 — Geração da implementação

`implement_model(model, use_float, backend="time-driven")` deve gerar:

- `snn_implementation.h`;
- `snn_implementation.cpp`;
- `neuron_params.h`;
- `quantization.h`.

Para o backend time-driven, deve gerar também `parallelism_manifest.json` com
o plano percentual resolvido por camada.

A função de topo gerada deve ter a interface:

```cpp
// Backend time-driven
void snn_to_hls(input_t (&input)[...], bit_t (&output)[...],
                bool reset_potentials);

// Backend event-driven, para self-events e Active List
void snn_to_hls(hls::stream<ed_spike_t>& input_stream,
                hls::stream<ed_spike_t>& output_stream,
                bool reset_potentials);
```

O gerador deve:

- declarar arrays com dimensões estáticas;
- serializar parâmetros NIR em `neuron_params.h`;
- preservar a ordem dos argumentos de cada primitiva;
- selecionar `time_driven.h` ou `event_driven.h` conforme o backend;
- manter estados neuronais entre chamadas;
- zerar todos os estados quando `reset_potentials` for verdadeiro;
- emitir erros antes da compilação quando uma camada não for suportada pelo
  backend selecionado.

Os nomes canônicos dos backends são `time-driven` e `event-driven`. O argumento
legado `use_event_driven` permanece aceito por compatibilidade.

### RF-05A — Estratégias CUBA-LIF event-driven

A estratégia padrão deve ser `active_list`. Na ausência de configuração
explícita, o gerador deve preservar a chamada CUBA-LIF analítica, a seleção
PWL e os arquivos produzidos por scripts anteriores.

Na estratégia `active_list`:

- leituras e acumulações sinápticas devem ocorrer somente para spikes de
  entrada;
- pesos de `Linear` e `Affine` devem ser convertidos para CSC orientado pela
  entrada, com `col_ptr`, `row_idx` e `values`; um spike deve ler somente os
  pesos não nulos de sua coluna, inclusive nas variantes recorrentes `Step`;
- cada neurônio deve aparecer no máximo uma vez na lista ativa, controlado por
  um vetor de flags indexado pelo ID;
- a entrada ponderada de um passo deve permanecer num acumulador `pending` até
  o watermark, em vez de ser incorporada ao estado e imediatamente decaída;
- `END_STEP` deve representar um lightweight tick e atualizar apenas os IDs
  ativos; `END_SAMPLE` deve executar o tick final antes de encerrar a amostra;
- spikes produzidos pelo tick devem ser emitidos antes do respectivo
  watermark;
- depois do tick, a lista deve ser compactada in-place, preservando somente os
  neurônios que ainda não atingiram o repouso;
- `reset_potentials` e o encerramento da amostra devem limpar estados, flags,
  entradas pendentes e quantidade de IDs ativos.

Definindo

```text
alpha = dt / tau_syn
beta  = dt / tau_mem
drive = beta * R * u
```

o gerador deve mover os ganhos constantes `alpha * beta * R * w_in` para o
caminho orientado a eventos. Para uma amplitude de entrada `a`, o incremento
pendente é:

```text
pending_drive += alpha * beta * R * w_in * a
```

No tick, a dinâmica equivalente deve ser calculada por:

```text
drive_next = (1 - alpha) * drive + pending_drive
v_next     = (1 - beta) * v + beta * v_leak + drive_next
```

Essa representação evita multiplicar `R` e `beta` para cada neurônio ativo e
mantém a entrada nova fora do termo de decaimento do mesmo passo. Aplicar
primeiro `u += alpha * w` e depois `(1 - alpha) * u` no mesmo tick seria
incorreto, pois acrescentaria à entrada um fator extra `(1 - alpha)`.

Os produtos por `alpha` e `beta` no caminho de tick devem usar aproximações
determinísticas por somas de no máximo quatro potências de dois:

```text
A_c(x) = soma(k=0..K-1, x >> deslocamento[k]), K <= 4
drive_next = drive - A_alpha(drive) + pending_drive
v_next     = v + A_beta(v_leak - v) + drive_next
```

Os termos e deslocamentos são constantes geradas a partir do modelo. A
quantização, o desempate da aproximação e a saturação devem ser determinísticos
e cobertos por testes. O caminho de tick não deve chamar a exponencial PWL,
o logaritmo, a busca do pico ou a bisseção usados por self-events.
Para manter o Euler explícito estável e representável apenas por right shifts,
o gerador deve exigir `0 < alpha <= 1` e `0 < beta <= 1` em cada neurônio.

Um neurônio pode ser removido somente quando não possuir entrada pendente e
tanto o módulo de `drive` quanto o módulo de `v - v_leak` forem menores ou
iguais ao valor configurado por `define_event_active_noise_threshold`. Eventos
positivos ou negativos cuja contribuição acumulada supere esse limiar devem
reativá-lo sem criar IDs duplicados.

`active_list` implementa exclusivamente o contrato
`event_cuba_lif_mode="discrete_compatible"`. A combinação com
Modos `continuous_physical` devem falhar antes da geração de arquivos, pois ticks
discretos não preservam os múltiplos disparos sub-step desse contrato. A
A aproximação PWL é fixa e é usada pelo caminho `active_list`.

### RF-06 — Grafos com múltiplas dependências e recorrência

Uma camada com mais de uma dependência deve receber a soma elemento a elemento
de todas elas. O frontend pode inserir primitivas `Merge` intermediárias.

- As dimensões e o tipo dos operandos devem coincidir.
- Uma dependência futura na ordem de travessia deve ser tratada como
  recorrente.
- Buffers recorrentes devem persistir entre passos e ser inicializados com
  zero.
- O resultado de um passo deve alimentar a dependência recorrente do passo
  seguinte, sem sobrescrever valores ainda necessários no passo atual.
- Recorrências inválidas ou ambíguas devem produzir erro descritivo.

### RF-07 — Testbench

`define_test_dataset(...)` deve aceitar:

- arquivo `.npz` com arrays `data` e `labels`; ou
- `torch.utils.data.TensorDataset` serializado pelo PyTorch.

Os dados devem ser convertidos em `tb_data/data.txt` e
`tb_data/targets.txt`. O formato deve respeitar `data_is_binary`.

`create_testbench(...)` deve:

- limitar `total_samples` ao total disponível;
- ajustar `batch_size` para um divisor válido de `total_samples`;
- suportar uma entrada fixa ou uma entrada diferente por passo;
- executar exatamente `step_count` passos por amostra;
- acumular spikes da saída e calcular a classe pelo maior acumulado;
- opcionalmente imprimir resultados de depuração;
- opcionalmente zerar estados entre inferências.

Parâmetros inválidos, conjunto de dados vazio ou shapes incompatíveis devem
produzir exceção clara antes da geração do testbench.

### RF-08 — Vitis HLS

O projeto deve oferecer:

- `run_csim(solution_name="sol")`;
- `run_synth(frequency_MHz, part="xc7z020clg400-1",
  solution_name="sol")`;
- `run_cosim(solution_name="sol", setup_only=False)`.

A frequência deve ser positiva. Falhas das ferramentas externas devem ser
propagadas ao chamador com comando, código de saída e contexto suficientes
para diagnóstico.

### RF-09 — Relatórios

Após uma síntese bem-sucedida:

- `get_synth_resource_usage` deve retornar BRAM, DSP, FF, LUT e URAM;
- `get_synth_performance_estimates` deve retornar ciclos e latência média.

O fluxo de potência deve permitir exportar o design, gerar atividade SAIF em
co-simulação ou pós-síntese e produzir relatório do Vivado. Arquivos ausentes
ou SAIF vazio devem ser rejeitados explicitamente.

## 6. Primitivas

### 6.1 Semântica requerida

| Primitiva | Operação |
|---|---|
| `Merge` | Soma elemento a elemento de tensores de mesmo shape. |
| `Flatten` | Converte `(C, H, W)` em vetor na ordem canal, linha e coluna. |
| `Linear` | `y = W x`. |
| `Affine` | `y = W x + b`. |
| `Scale` | `y[i] = x[i] * scale[i]`, com broadcasting NIR válido. |
| `Conv1d` | Convolução 1D com stride, padding, dilation, groups e bias opcional. |
| `Conv2d` | Convolução 2D com stride, padding, dilation, groups e bias opcional. |
| `SumPool2d` | Soma dos valores válidos da janela. Padding contribui com zero. |
| `AvgPool2d` | Média dos valores da janela, com regra de padding compatível com NIR. |
| `I` | Integrador sem vazamento. |
| `IF` | Integrador com threshold, spike e reset. |
| `LI` | Integrador com vazamento, sem emissão de spike. |
| `LIF` | Integrador com vazamento, threshold, spike e reset. |
| `CubaLI` | Dinâmica de corrente sináptica e membrana sem spike. |
| `CubaLIF` | `CubaLI` com threshold, spike e reset para valor ou por subtração. |

As equações discretas devem usar Euler explícito e um `dt` configurável. O
valor de `dt` não deve ficar implícito na primitiva ou ser alterado pelo
backend.

### 6.2 Matriz de suporte atual

Os estados abaixo descrevem o repositório na elaboração desta especificação:

- **Integrado**: participa do fluxo principal de geração.
- **Parcial**: há configuração ou código da primitiva, mas falta integração,
  cobertura dimensional ou validação de ponta a ponta.
- **Futuro**: ainda precisa ser implementado no fluxo principal.

| Primitiva | Frontend NIR | Backend time-driven | Event-driven | Estado geral |
|---|---:|---:|---:|---|
| `Merge` | Sim | Sim | Sim | Recorrência integrada no event-driven; pendente no denso. |
| `Flatten` | Sim | Sim | Sim | Integrado. |
| `Linear` | Sim | Sim | Sim | Integrado. |
| `Affine` | Sim | Sim | Sim | Integrado. |
| `Scale` | Sim | 1D, 2D e 3D | Não | Integrado no backend time-driven. |
| `Conv1d` | Sim | Sim | Não | Integrado no backend time-driven. |
| `Conv2d` | Sim | Sim | Sim | Integrado nos dois backends. |
| `SumPool2d` | Sim | Sim | Sim | Integrado nos dois backends. |
| `AvgPool2d` | Sim | Sim | Não | Integrado no backend time-driven. |
| `I` | Sim | Código não integrado | Não | Futuro. |
| `IF` | Sim | Parcial | 1D e 3D | Parcial. |
| `LI` | Sim | Código não integrado | Não | Futuro. |
| `LIF` | Sim | Código não integrado | 1D e 3D | Parcial. |
| `CubaLI` | Sim | Código não integrado | Não | Futuro. |
| `CubaLIF` | Sim | 1D | 1D, self-events e Active List | Parcial: ampliar dimensões e validar equivalência. |

## 7. Funções futuras

As funções futuras devem ser implementadas na ordem abaixo. Uma função só
deve ser marcada como concluída após cumprir os critérios da seção 9.

### P0 — Correção e fechamento do backend time-driven

1. **Validação antecipada do backend**: impedir geração de C++ inválido para
   primitivas ou shapes não suportados.
2. **Merge recorrente**: corrigir o encadeamento de múltiplas entradas e
   garantir atraso de exatamente um passo nas arestas recorrentes.
3. **Estado e `dt`**: tornar `dt` configurável e padronizar criação, reset e
   tipo dos estados de `I`, `IF`, `LI`, `LIF`, `CubaLI` e `CubaLIF`.
4. **Modelos neuronais densos**: integrar `I`, `IF`, `LI`, `LIF` e `CubaLI`
   ao gerador nas formas 1D, 2D e 3D.
5. **CubaLIF multidimensional**: suportar 2D e 3D, reset por valor e por
   subtração.

### P1 — Backend orientado a eventos — concluído em 2026-07-17

1. [x] Implementar `Conv2d` orientada a eventos.
2. [x] Implementar `SumPool2d` orientada a eventos.
3. [x] Definir suporte de `IF`, `LIF` e `CubaLIF`.
4. [x] Garantir propagação correta de `END_STEP` e `END_SAMPLE` por todas as
   primitivas.
5. [x] Validar equivalência do backend event-driven com o backend time-driven para a
   mesma sequência de eventos.

O gerador event-driven suporta grafos sequenciais, ramificados e recorrentes
compostos por `Linear`, `Affine`, `Flatten`, `Conv2d`, `SumPool2d`, `IF`, `LIF`
e `CubaLIF`. Componentes fortemente conexos são identificados na representação
intermediária; arestas de retorno são convertidas em streams persistentes com
atraso de um passo, e fan-in/fan-out são materializados com `Merge`/`Split`.

Para CUBA-LIF, `active_list` mantém a dinâmica discreta por ticks e
mantém a leitura sináptica orientada a eventos, mas avança somente neurônios
ativos nos watermarks discretos. As estratégias pertencem ao mesmo backend e
não alteram a interface externa por streams.

### P2 — Operadores matemáticos pendentes — concluído em 2026-07-17

1. [x] **Conv1d**: implementar grupos, dilation, padding e bias opcional.
2. [x] **AvgPool2d**: definir e testar a contagem do divisor nas bordas.
3. [x] **Scale**: adicionar configuração NIR, broadcasting estático e primitiva
   para shapes 1D, 2D e 3D.
4. [x] **Bias opcional em Conv2d**: a ausência de bias não deve gerar parâmetro ou
   acesso inválido.
5. [x] **Padding simbólico**: normalizar `same` e `valid` para valores estáticos ou
   rejeitar combinações que não possam ser resolvidas em compilação.


### P3 — Robustez e experiência de uso

1. Validar quantização, shapes, grupos, strides e caminhos antes de gerar
   arquivos.
2. Substituir mensagens apenas impressas por exceções específicas.
3. Tornar a geração de arquivos atômica para não deixar saída parcial em caso
   de erro.
4. Disponibilizar uma interface de linha de comando para o fluxo principal.
5. Criar testes automatizados que não dependam do Vitis e uma suíte opcional
   de integração com Vitis/Vivado.

## 8. Requisitos não funcionais

### RNF-01 — Compatibilidade HLS

- O C++ gerado deve usar estruturas sintetizáveis pelo Vitis HLS.
- Alocação dinâmica, recursão em C++ e shapes dinâmicos não são permitidos no
  caminho sintetizável.
- Diretivas HLS devem ser justificadas por síntese ou medição.

### RNF-02 — Determinismo

Para o mesmo NIR, metadados, opções e versões de dependências, os arquivos
gerados devem ser semanticamente idênticos e produzir os mesmos resultados.

### RNF-03 — Precisão numérica

- O modo `float` é a referência do backend C++.
- Testes em ponto fixo devem declarar sua tolerância e configuração de bits.
- Overflow, arredondamento e saturação devem ser definidos explicitamente.
- Aproximações shift-add devem registrar os termos binários escolhidos e usar
  no máximo quatro potências de dois por coeficiente `alpha` ou `beta`.
- Divergências de spike não podem ser avaliadas apenas por erro médio; a
  sequência e o instante do spike também devem ser comparados.

### RNF-04 — Portabilidade

O frontend Python deve funcionar em Linux e Windows. A execução de HLS depende
de uma instalação compatível do Vitis/Vivado e pode exigir configuração de
ambiente específica da plataforma.

### RNF-05 — Documentação

Toda alteração funcional deve criar ou atualizar um registro em `docs/` com:

- requisito atendido;
- arquivos modificados;
- decisão de implementação;
- comandos de teste executados e resultados;
- limitações conhecidas e próximos passos.

## 9. Critérios de aceite

Uma primitiva ou função é considerada concluída quando todos os itens
aplicáveis forem atendidos:

1. o nó NIR é convertido para uma configuração válida;
2. shapes e parâmetros são validados antes da geração;
3. o C++ gerado compila em modo `float` com um compilador C++ de teste;
4. a C-simulação do Vitis compila e executa, quando o Vitis está disponível;
5. a saída coincide com uma referência NumPy, PyTorch ou NIR para casos
   nominais e de borda;
6. os testes cobrem pelo menos 1D e todas as dimensionalidades declaradas;
7. estado e reset são testados em múltiplos passos para modelos neuronais;
8. ponto fixo é testado com tolerância e quantização documentadas;
9. documentação e matriz de suporte são atualizadas;
10. nenhum teste anteriormente aprovado sofre regressão.

Para camadas que emitem spikes, a equivalência deve comparar o tensor de
spikes em cada passo. Para camadas com estado, o estado interno final também
deve ser comparado quando acessível no teste.

## 10. Fluxo de uso esperado

```python
from neuro_hls import NeuroHls

hls = NeuroHls("build/minha_rede")
model = hls.read_nir_file(
    "modelo.nir",
    metadata_file_path="modelo.metadata.json",
)

model.define_input_quantization(16, 8)
model.define_weight_quantization(16, 8)
model.define_potential_quantization(24, 8)

# Alternativa event-driven com Active List:
# model.define_event_cuba_lif_strategy("active_list")
# model.define_event_active_noise_threshold(1e-6)
# hls.implement_model(model, use_float=False, backend="event-driven")

hls.implement_model(model, use_float=False, backend="time-driven")
hls.define_test_dataset(
    "dataset.npz",
    data_is_binary=True,
    step_count=30,
    different_sample_per_step=True,
)
hls.create_testbench(
    total_samples=1000,
    batch_size=10,
    reset_potentials=True,
    debug_mode=False,
)

hls.run_csim()
hls.run_synth(frequency_MHz=100)

print(hls.get_synth_resource_usage())
print(hls.get_synth_performance_estimates())
```

Esse exemplo representa a API pretendida. A execução completa depende de o
modelo utilizar apenas primitivas suportadas pelo backend selecionado e de as
ferramentas AMD estarem configuradas.

As variantes SRNN Active List de referência devem ser geradas pelo script
reproduzível, que também atualiza a cópia local do backend:

```bash
.venv/bin/python SRNN_tests/generate_active_list.py
```

## 11. Dependências

### Python

- Python 3;
- NumPy;
- PyTorch, para conjuntos de dados serializados;
- biblioteca `nir`.

### Síntese

- Vitis HLS com o comando `vitis-run`;
- Vivado para exportação, simulação RTL, SAIF e potência;
- headers HLS, incluindo `ap_int.h` e `ap_fixed.h`.

As versões homologadas devem ser registradas em documentação de instalação ou
em arquivo de dependências versionado.

## 12. Limitações conhecidas

- A cobertura automatizada do pacote principal ainda é insuficiente.
- Nem toda primitiva reconhecida pelo frontend possui chamada compatível no
  backend time-driven.
- A recorrência orientada a eventos pressupõe atraso de um passo em cada
  aresta de retorno identificada pela transformação do grafo.
- `CubaLIF` está integrada apenas para vetores 1D.
- A estratégia CUBA-LIF `active_list` suporta apenas a dinâmica discreta;
  `continuous_physical` não é suportado.
- A estratégia Active List exige `0 < dt/tau_syn <= 1` e
  `0 < dt/tau_mem <= 1` para todos os neurônios.
- O erro introduzido pela decomposição shift-add de `alpha` e `beta` deve ser
  considerado ao comparar timestamps e acurácia com o backend time-driven.
- O passo `dt` de `CubaLIF` está fixado na configuração atual.
- A geração atual pode produzir C++ inválido em vez de rejeitar cedo algumas
  combinações não suportadas.
- Alguns fluxos de potência usam part e caminhos de ferramenta específicos e
  ainda precisam ser parametrizados.
- A recorrência do backend time-driven ainda requer correção e testes adicionais.

## 13. Política de evolução

- Alterações incompatíveis na API pública devem ser registradas em `docs/`.
- Uma primitiva não deve ser marcada como suportada apenas porque existe um
  header experimental; é necessário suporte de ponta a ponta.
- Novas funções devem incluir testes de erro, não apenas casos de sucesso.
- O `README.md` deve apresentar um resumo para usuários, enquanto este arquivo
  permanece como fonte de requisitos e critérios de aceite.
