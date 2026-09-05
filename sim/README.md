# Ambiente de simulação NeuroHLS

Este diretório contém um fluxo executável para projetos já gerados pelo
NeuroHLS. A entrada é uma pasta clássica NeuroHLS; arquivos `.nir`, datasets
avulsos e RTL exportado isoladamente são rejeitados.

O fluxo cria uma cópia da entrada em `sim/runs/`, para que Vitis e Vivado nunca
modifiquem o projeto original. Em seguida executa CSim, síntese HLS, setup de
co-simulação, exportação IP pelo Vitis, síntese OOC, simulação funcional
pós-síntese, captura SAIF e relatório de potência anotado.

## Pré-requisitos

- Python 3.9 ou superior;
- PyYAML apenas se for usado `--environment`;
- Vitis HLS e Vivado no `PATH`, ou `tools.settings_script` configurado;
- uma licença que cubra CSim, síntese HLS, exportação, XSim e Vivado.

O projeto precisa ter, no mínimo, os arquivos `0_create_project.tcl` a
`3_cosim.tcl`, `snn_implementation.cpp`, `snn_implementation.h`,
`testbench.cpp`, `neuron_params.h`, `quantization.h`,
`neuro_hls_functions/` e `tb_data/{data,targets}.txt`.

## Uso

Primeiro valide a entrada:

```bash
python -m sim validate --project z_test_event_driven_scnn
```

Faça o preflight, que cria uma run e confirma ferramentas e contrato de
entrada:

```bash
python -m sim preflight --project z_test_event_driven_scnn
```

Para executar o fluxo completo:

```bash
python -m sim run \
  --project z_test_event_driven_scnn \
  --environment sim/environment.example.yaml
```

Use outro diretório no argumento `--project` para trocar a rede. As opções
`--part`, `--frequency-mhz`, `--clock-name` e
`--saif-min-match-percent` substituem valores do YAML sem alterar o arquivo.

Para limitar uma execução ou retomá-la:

```bash
python -m sim run --project z_test_event_driven_scnn --to csim
python -m sim resume --run sim/runs/z_test_event_driven_scnn/<run-id>
python -m sim status --run sim/runs/z_test_event_driven_scnn/<run-id>
```

`--dry-run` valida a entrada e cria um plano de etapas, sem acionar Vitis ou
Vivado.

### Etapas executáveis

As etapas disponíveis, executadas nesta ordem, são:

1. `prepare`
2. `vitis-project`
3. `csim`
4. `hls-synth`
5. `cosim-setup`
6. `export`
7. `vivado-synth`
8. `post-synth-sim`
9. `power`

### Origem da atividade para a potência

`power.activity_source` decide como a potência é estimada:

| valor | como estima | custo | etapas necessárias |
| --- | --- | --- | --- |
| `vectorless` (padrão) | taxas de transição padrão declaradas na configuração | minutos | dispensa `cosim-setup` e `post-synth-sim` |
| `saif` | atividade medida na simulação de gate pós-síntese | horas a dezenas de horas | exige as duas |

O padrão é `vectorless` porque a simulação que produz o SAIF domina o tempo do
fluxo: nas execuções deste repositório ela levou de 5 a mais de 30 horas por
design, enquanto todas as demais etapas somam poucos minutos. No modo
`vectorless`, `cosim-setup` e `post-synth-sim` são marcadas `skipped` com o
motivo registrado, e `--from power` funciona sem que elas tenham sido
executadas.

Uma estimativa vectorless usa taxas de transição padrão em vez da atividade
real do workload, então ela nunca é definitiva: o resumo registra
`activity_source: vectorless`, `provisional: true` e `saif_coverage_passed:
null`, já que não existe anotação cuja cobertura possa ser medida. Para o
número anotado, use:

```bash
python -m sim run --project meu_projeto \
  --environment ambiente.yaml   # com power.activity_source: saif
```

É possível selecionar uma etapa final ou um intervalo:

```bash
python -m sim run --project z_test_event_driven_scnn --to vivado-synth
python -m sim run --project z_test_event_driven_scnn \
  --from vivado-synth --to power
```

### Dataset por etapa

O testbench principal (`testbench.cpp` e `tb_data/`) é usado integralmente no
CSim. Se o projeto também contiver o bundle abaixo, o pipeline o ativa antes de
`cosim-setup`; assim, CoSim e a simulação funcional pós-síntese usam apenas os
medoids:

```text
tb_medoids/
├── testbench.cpp
├── data.txt
└── targets.txt
```

A troca ocorre somente na cópia isolada do run. O projeto de entrada permanece
inalterado.

## Resultados

Cada run contém:

```text
sim/runs/<projeto>/<timestamp>-<hash>/
├── project/                 # cópia isolada da entrada
├── logs/                    # stdout/stderr de cada ferramenta
├── 40_vivado_synth/         # DCP, netlist, timing e utilização
├── 50_post_synth_sim/       # PRJ/autotb pós-síntese gerados
├── 60_activity/             # SAIF pós-síntese
├── 70_power/                # relatório do Vivado e nets não anotadas
└── reports/
    ├── preflight.json
    ├── activity_summary.json
  ├── utilization_summary.json
    ├── power_summary.json
    ├── summary.json
    └── summary.md
```

### Métricas automáticas

Quando o testbench clássico expõe `TOTAL_SAMPLES`, `BATCH_SIZE` e
`STEP_COUNT`, os resumos JSON e Markdown registram automaticamente:

- amostras efetivamente executadas, batches, steps por amostra e steps lógicos
  totais;
- duração lógica da janela SAIF em segundos e em unidade legível;
- latência média amortizada por step e por amostra;
- estimativa HLS separada do uso pós-síntese do Vivado OOC;
- utilização pós-síntese com capacidade total do FPGA, uso por recurso e
  percentual recalculado para validação;
- potência total, dinâmica e estática, confiança do Vivado e cobertura SAIF;
- energia total da captura e energia média por step e por amostra.

As métricas derivadas usam:

```text
steps_totais       = amostras_executadas × steps_por_amostra
latência/step      = duração_janela / steps_totais
latência/amostra   = duração_janela / amostras_executadas
energia            = potência_média × duração_janela
energia/step       = energia / steps_totais
energia/amostra    = energia / amostras_executadas
```

A `duração_janela` depende de `power.activity_source`. Com `saif`, é a janela
de captura medida na simulação pós-síntese. Sem SAIF não existe captura, e a
janela vem da latência relatada pelo HLS:

```text
duração_janela = latência_HLS × período_do_clock × steps_lógicos_totais
```

`latency_definition` e `energy_definition` registram qual das duas origens foi
usada, porque o número de energia só é interpretável junto com ela. Nas runs em
que as duas estão disponíveis, a janela analítica reproduz a medida com erro de
cerca de 0,1%.

Os steps são lógicos: em um backend event-driven, a quantidade de transações
do DUT pode ser diferente. A latência é amortizada sobre toda a janela SAIF,
incluindo inicialização, intervalos e finalização, e não representa uma
medição isolada de `ap_start` até `ap_done`.

O relatório de potência é interpretado e persistido antes da aplicação do
limite de cobertura. Assim, se `Design Nets Matched` reprovar, os números
permanecem disponíveis para diagnóstico, mas recebem
`saif_coverage_passed: false`, `provisional: true` e a etapa `power` continua
em `failed`. A presença do artefato de potência, isoladamente, não significa
que o resultado foi aceito; consumidores devem verificar
`quality_accepted` e o estado da etapa.

O resumo consolida recursos em duas camadas distintas: a estimativa HLS,
obtida do `csynth.xml`, e a utilização pós-síntese do Vivado OOC, obtida do
relatório global sem `-hierarchical`. Antes da síntese Vivado, o ambiente
mostra apenas a estimativa HLS ou indica que a utilização pós-síntese ainda
não está disponível.

O fluxo falha deliberadamente se a simulação pós-síntese falhar, se o SAIF for
vazio, tiver duração zero ou não tiver transições, ou se a cobertura
`Design Nets Matched` ficar abaixo do limite configurado. Isso evita aceitar
como válido um número de potência baseado em atividade insuficiente.

## Limites da versão inicial

- É suportado o layout clássico de pasta NeuroHLS, não o layout de componentes
  Vitis `hls_config.cfg`.
- O único perfil implementado é `post-synth`, com netlist funcional
  pós-síntese. O perfil pós-route com SDF ainda não está implementado.
- A qualidade do valor de potência continua dependente de estímulos
  representativos e de uma boa cobertura SAIF; o relatório registra essa
  cobertura para auditoria.
