# Especificação do ambiente de simulação e estimativa de potência NeuroHLS

**Status:** MVP `post-synth` implementado; integração completa com ferramentas ainda em validação  
**Versão:** 0.4  
**Data:** 2026-07-26  
**Diretório:** `sim/`

## 1. Objetivo

Este documento especifica um ambiente reproduzível para validar e caracterizar
componentes de redes neurais gerados pelo NeuroHLS. O usuário deve conseguir
trocar a rede neural, selecionar um perfil de execução e iniciar todo o fluxo
sem editar scripts TCL:

1. importar uma pasta de projeto já gerada pelo NeuroHLS;
2. executar CSim e síntese HLS;
3. exportar o RTL com Vitis HLS;
4. sintetizar o componente no Vivado;
5. executar simulação funcional pós-síntese;
6. capturar a atividade de comutação em um arquivo SAIF;
7. anotar o SAIF no design sintetizado;
8. gerar relatórios de potência, utilização, temporização e correção funcional.

O ambiente ficará dentro deste repositório, mas seus artefatos de execução
serão isolados do código-fonte do NeuroHLS e dos componentes de entrada.

## 2. Resultado esperado para o usuário

O fluxo completo deverá receber diretamente a pasta do projeto gerado pelo
NeuroHLS:

```bash
python -m sim run \
  --project z_test_event_driven_scnn \
  --profile post-synth
```

Trocar a rede consistirá em informar outra pasta de projeto:

```bash
python -m sim run \
  --project z_test_event_driven_srnn \
  --profile post-synth
```

Nenhum nome de top, part FPGA, período de clock, caminho de solução, hierarquia
SAIF ou diretório de saída deverá precisar ser alterado manualmente em TCL.

## 3. Escopo

### 3.1 Incluído

- componentes time-driven e event-driven gerados pelo NeuroHLS;
- entrada exclusivamente a partir de uma pasta de projeto já gerada pelo
  NeuroHLS, contendo fontes, testbench e dados de teste;
- CSim, síntese e configuração de co-simulação do Vitis HLS;
- exportação do RTL/IP;
- síntese out-of-context no Vivado;
- simulação funcional pós-síntese com XSim;
- verificação das saídas da netlist sintetizada;
- captura e validação de SAIF;
- estimativa de potência com atividade anotada;
- retomada do fluxo a partir de uma etapa já concluída;
- histórico imutável de execuções e relatórios consolidados;
- perfil opcional pós-implementação para uma estimativa de potência mais
  próxima do hardware físico.

### 3.2 Fora do escopo inicial

- treinamento ou retreinamento de redes;
- programação e execução em placa;
- drivers de host, PCIe, AXI DMA ou integração com processador;
- estimativa térmica da placa completa;
- potência de componentes externos ao FPGA;
- fechamento automático de timing por exploração de diretivas;
- síntese distribuída em várias máquinas;
- leitura de modelos `.nir`;
- geração da rede neural pelo ambiente de simulação;
- aceitação de RTL ou IP exportado sem o projeto NeuroHLS que o originou;
- seleção de um dataset externo diferente daquele já materializado no projeto.

## 4. Premissas e precisão da estimativa

A presença de um SAIF não torna, por si só, a estimativa de potência precisa.
A qualidade do resultado depende de quatro fatores:

1. estímulos representativos da aplicação real;
2. janela de captura que exclua inicialização não representativa;
3. correspondência entre a hierarquia do SAIF e a hierarquia sintetizada;
4. qualidade das informações físicas usadas pelo Vivado.

O perfil obrigatório `post-synth` usará a netlist funcional pós-síntese. Ele é
adequado para validação, comparação entre arquiteturas e estimativas iniciais,
mas ainda usa estimativas para parte do roteamento.

O perfil `power-accurate`, previsto como evolução, deverá executar
place-and-route, simulação temporal pós-implementação com SDF e leitura do SAIF
no checkpoint implementado. Esse será o perfil recomendado para números finais
de potência. Mesmo nesse perfil, o resultado continuará sendo uma estimativa e
deverá informar cobertura de anotação, condições de operação e confiança do
Vivado.

## 5. Princípios de arquitetura

### 5.1 Separação de responsabilidades

O ambiente será dividido em:

- **componente:** pasta de projeto já gerada pelo NeuroHLS, incluindo fontes,
  parâmetros, testbench e dados de teste;
- **plataforma:** FPGA, clock, versão das ferramentas e condições de potência;
- **pipeline:** implementação das etapas e scripts reutilizáveis;
- **run:** cópia imutável das entradas, logs, intermediários e resultados.

Scripts de pipeline não poderão conter valores específicos de uma rede.

### 5.2 Fonte única de configuração

Arquivos YAML validados por schema serão a fonte de configuração. Scripts
Python e TCL receberão parâmetros gerados a partir desses arquivos. Não haverá
uma segunda configuração divergente escondida em notebooks ou scripts.

### 5.3 Falha explícita

Uma etapa somente será marcada como concluída quando:

- o processo externo terminar com código zero;
- todos os artefatos obrigatórios existirem e não estiverem vazios;
- as validações específicas da etapa passarem.

Mensagens impressas contendo erro não poderão ser tratadas como sucesso. O
orquestrador deverá propagar comando, código de saída, etapa, log e sugestão de
correção.

### 5.4 Reprodutibilidade

Cada execução registrará:

- commit Git e estado de modificações locais;
- versão do NeuroHLS;
- versões completas de Python, Vitis HLS, Vivado, XSim e sistema operacional;
- hash agregado do projeto de entrada e hashes de C++, headers, testbench,
  dados de teste e RTL;
- configuração resolvida;
- comandos executados;
- part, clock, seeds, diretivas e condições de potência;
- data, duração e status de cada etapa.

## 6. Arquitetura lógica

```text
projeto NeuroHLS       environment.yaml
       |                       |
       +-----------+-----------+
                  |
           validação/preflight
                  |
          cópia isolada do projeto
                  |
          CSim -> síntese HLS
                  |
        setup de co-sim -> export RTL/IP
                  |
          síntese OOC no Vivado
                  |
       simulação funcional pós-síntese
                  |
         comparação + captura SAIF
                  |
       read_saif no checkpoint correto
                  |
        power/timing/utilização/resumo
```

O orquestrador é implementado em Python. TCL é usado apenas para ações
das ferramentas AMD. A comunicação Python–TCL ocorrerá por argumentos ou por
um arquivo de configuração resolvido gerado dentro do diretório da execução.

## 7. Estrutura de diretórios implementada

```text
sim/
├── SPEC.md
├── README.md
├── __main__.py
├── cli.py
├── config.py
├── project.py
├── pipeline.py
├── tcl.py
├── utils.py
├── environment.example.yaml
├── tests/
│   └── test_environment.py
└── runs/
    └── .gitkeep
```

Os scripts TCL são gerados por execução dentro de `runs/<id>/`, recebendo
part, clock, top, caminhos e hierarquia SAIF resolvidos. Isso evita manter TCL
específico de uma rede em `sim/`.

O diretório `runs/` deverá ser ignorado pelo Git, exceto por `.gitkeep`.
Relatórios que o usuário decidir preservar poderão ser copiados para outro
diretório versionado.

## 8. Contrato do projeto de entrada

### 8.1 Entrada única

O argumento `--project` aceitará somente o caminho de um diretório gerado pelo
NeuroHLS. Não haverá `source.kind` nem modos alternativos de entrada.

Os casos abaixo deverão ser rejeitados antes da execução das ferramentas:

- caminho para um arquivo `.nir`;
- caminho para um dataset;
- arquivo `export.zip`;
- diretório contendo somente RTL ou um IP exportado;
- diretório comum que não satisfaça a estrutura mínima NeuroHLS.

Arquivos `.nir` eventualmente armazenados dentro da pasta do projeto não serão
lidos nem usados pelo ambiente. A entrada funcional será sempre o C++,
testbench e dados já materializados pelo NeuroHLS.

### 8.2 Estrutura mínima

Um projeto válido deverá conter:

```text
<project>/
├── 0_create_project.tcl
├── 1_csim.tcl
├── 2_synth.tcl
├── 3_cosim.tcl
├── snn_implementation.cpp
├── snn_implementation.h
├── testbench.cpp
├── neuron_params.h
├── quantization.h
├── neuro_hls_functions/
└── tb_data/
    ├── data.txt
    └── targets.txt
```

Headers adicionais gerados para operadores da rede serão aceitos e copiados.
O subdiretório de projeto Vitis, normalmente `vitis_proj/`, poderá existir ou
ser criado durante a execução.

### 8.3 Identidade e descoberta

- O identificador padrão do componente será derivado do nome da pasta.
- A pasta de entrada será tratada como somente leitura.
- O ambiente criará uma cópia isolada antes de alterar scripts ou gerar
  artefatos.
- O hash da entrada incluirá todos os arquivos necessários para reproduzir
  CSim e síntese, inclusive `tb_data`.
- Arquivos temporários, logs e resultados antigos de ferramentas não
  participarão do hash funcional.
- O top, shapes, tipos e interface deverão ser descobertos dos fontes e
  metadados produzidos pelo HLS.
- O top HLS padrão será `snn_to_hls`, mas nenhum script poderá presumir esse
  nome sem receber a configuração.
- A interface efetiva do RTL, incluindo larguras, dimensões e protocolo de
  controle, será extraída dos metadados do HLS e registrada.
- A rede poderá mudar shapes sem exigir alterações manuais no ambiente.
- Recorrência e estados internos serão responsabilidade do componente; o
  ambiente controlará apenas reset, transações e estímulos.

## 9. Contrato da plataforma

Exemplo de `environment.yaml`:

```yaml
schema_version: 1

tools:
  expected_version: "2025.2"
  vitis_run: vitis-run
  vivado: vivado
  xelab: xelab
  xsim: xsim
  settings_script: null

target:
  part: xcu250-figd2104-2L-e
  clock:
    name: ap_clk
    frequency_mhz: 150
    uncertainty_ns: 0.2

execution:
  jobs: 1
  keep_intermediates: true
  reuse_cache: true
  timeout_minutes: 180

simulation:
  timescale: 1ns/1ps
  profile: post-synth
  verify_outputs: true
  activity:
    warmup_samples: 1
    measured_samples: all_remaining
    capture_scope: dut

power:
  process: typical
  ambient_temperature_c: 25
  saif_min_match_percent: 50
  fail_on_low_confidence: true
  report_hierarchy: true
```

O schema deverá rejeitar frequência não positiva, part vazio, quantidade de
jobs inválida, janela de atividade incoerente e versões de schema
desconhecidas.

## 10. Interface estável e adaptação do RTL

As portas geradas pelo HLS mudam com os shapes da rede. Portanto, "trocar
somente o componente" não significa que todas as redes terão um RTL idêntico.
O ambiente deverá oferecer uma camada de adaptação gerada automaticamente.

Essa camada deverá:

- identificar as portas e larguras a partir do RTL/metadados HLS;
- instanciar o top configurado;
- dirigir clock, reset e protocolo `ap_ctrl_hs`;
- carregar entradas e referências em formatos independentes do testbench HLS;
- capturar saídas e status de conclusão;
- expor marcadores de início e fim da janela de atividade;
- preservar uma instância DUT com nome estável para mapeamento do SAIF.

No MVP, o testbench criado pelo Vitis HLS poderá ser reutilizado para reduzir o
tempo de implementação. Nesse caso, o parser não poderá depender de nomes
fixos como `apatb_snn_to_hls_top` e `AESL_inst_snn_to_hls`: os nomes deverão
ser descobertos ou construídos a partir do top. A camada estável deverá
substituir progressivamente essa dependência do autotestbench interno do HLS.

## 11. Pipeline

### 11.1 Etapa 00 — criação da execução

O comando `run` deverá:

1. validar os YAMLs contra seus schemas;
2. combinar o projeto, a plataforma e o perfil;
3. calcular um identificador a partir de data e hash do projeto;
4. criar um diretório exclusivo;
5. salvar a configuração resolvida, o inventário do projeto e seus hashes;
6. impedir sobrescrita de uma execução anterior.

Estrutura de uma execução:

```text
runs/<component-id>/<timestamp>-<hash>/
├── run.yaml
├── status.json
├── 10_neurohls/
├── 20_hls/
├── 30_export/
├── 40_vivado_synth/
├── 50_post_synth_sim/
├── 60_activity/
├── 70_power/
├── reports/
└── logs/
```

### 11.2 Etapa 01 — preflight

Antes de iniciar trabalho caro, o ambiente verificará:

- presença e versão das ferramentas;
- licença utilizável para as etapas solicitadas;
- disponibilidade do part no Vivado;
- acesso de leitura ao projeto e escrita à pasta de runs;
- presença e consistência dos fontes, headers, testbench e `tb_data`;
- espaço livre mínimo configurável;
- ausência de caminhos hardcoded de outra máquina;
- consistência entre frequência HLS e clock Vivado;
- suporte do perfil pela versão instalada das ferramentas.

O relatório `preflight.json` listará cada verificação e seu resultado.

### 11.3 Etapa 10 — importação do projeto NeuroHLS

O ambiente validará o projeto informado, calculará seus hashes e copiará o
conteúdo necessário para a execução. Todas as ferramentas trabalharão sobre
essa cópia isolada.

O ambiente não chamará `read_nir_file`, `implement_model`,
`define_test_dataset` ou `create_testbench`. A geração da rede e do testbench
deverá ter sido concluída pelo usuário antes de iniciar este fluxo.

Artefatos obrigatórios:

- fontes C++ geradas;
- headers e parâmetros;
- testbench e dados convertidos;
- manifesto resolvido com arquivos, top, shapes e tipos descobertos;
- hashes de todos os arquivos usados pela síntese.

### 11.4 Etapa 20 — CSim

A CSim será habilitada por padrão e funcionará como golden funcional do fluxo.
Serão registrados:

- código de saída;
- acurácia e/ou comparação por amostra;
- total de amostras executadas;
- duração;
- stdout e stderr completos.

Exceção de ponto flutuante, timeout, divergência ou ausência do resumo final
deverá interromper o perfil estrito.

### 11.5 Etapa 21 — síntese HLS

A síntese HLS usará o part e período da configuração resolvida. Ela deverá
produzir:

- RTL sintetizável;
- XML e texto de síntese;
- latência estimada;
- intervalo de iniciação;
- utilização estimada de BRAM, DSP, FF, LUT e URAM;
- constraints e metadados da interface.

A exportação não poderá continuar se a solução tiver sido sintetizada para
outro part ou outro período sem que isso seja explicitamente permitido.

### 11.6 Etapa 30 — co-sim setup e exportação

O ambiente executará o setup da co-simulação para gerar os vetores e arquivos
auxiliares necessários, seguido da exportação do componente.

Os artefatos exportados deverão ser descobertos por conteúdo/metadados, e não
por um único caminho específico de uma versão do Vitis. O ambiente registrará:

- lista de HDL e includes;
- top exportado;
- constraints;
- pacote IP ou `export.zip`, quando disponível;
- arquivos do testbench e vetores;
- hash agregado do RTL.

### 11.7 Etapa 40 — síntese out-of-context no Vivado

O Vivado deverá:

- criar um projeto isolado ou in-memory para a execução;
- ler todo o RTL exportado e diretórios de include;
- aplicar o clock da plataforma;
- sintetizar o top em modo out-of-context;
- manter hierarquia suficiente para anotação de atividade;
- emitir checkpoint e netlist funcional;
- gerar relatórios de utilização, timing e metodologia.

Artefatos mínimos:

- `post_synth.dcp`;
- `post_synth_netlist.v`;
- `post_synth.sdf`, se produzido e aplicável;
- `utilization_post_synth.rpt`;
- `utilization_device_post_synth.rpt`;
- `timing_post_synth.rpt`;
- `methodology_post_synth.rpt`.
- `reports/utilization_summary.json`.

O primeiro relatório de utilização é hierárquico e preserva a compatibilidade
com o fluxo anterior. O segundo é global, sem `-hierarchical`, e serve como
fonte principal para o resumo pós-síntese, com `Used`, `Available` e `Util%`
normalizados por recurso.

O part não poderá estar fixo no TCL. O valor usado deverá constar no log e no
manifesto da execução.

### 11.8 Etapa 50 — simulação pós-síntese

A simulação funcional usará a netlist recém-gerada, as bibliotecas corretas do
Vivado e exatamente o mesmo conjunto de estímulos da validação funcional.

O testbench deverá:

- gerar clock e reset conforme a configuração;
- obedecer ao protocolo do componente;
- aguardar a conclusão sem usar atrasos frágeis;
- impedir divisão por zero no cálculo de progresso ou batches;
- comparar todas as saídas relevantes com a referência;
- produzir resultado legível por máquina;
- encerrar com falha em caso de mismatch, timeout ou saída desconhecida;
- emitir marcadores da janela válida de atividade.

O SAIF somente será aceito se a simulação funcional também passar. A opção de
gerar SAIF com mismatch deverá existir apenas como modo de diagnóstico e ser
marcada claramente como não válida para relatório oficial.

### 11.9 Etapa 60 — captura e validação SAIF

A captura padrão deverá incluir apenas a hierarquia do DUT, excluindo o
testbench. A janela deverá:

1. começar após reset e aquecimento configurado;
2. conter um número conhecido de inferências completas;
3. terminar antes da finalização administrativa do testbench.

O ambiente deverá registrar no `activity_summary.json`:

- arquivo, tamanho e hash do SAIF;
- timescale;
- duração total da captura em unidades SAIF e normalizada em segundos;
- amostras declaradas e efetivamente executadas, tamanho/quantidade de batches,
  steps lógicos por amostra e steps lógicos totais;
- scope de captura;
- frequência efetiva;
- proporção entre aquecimento e medição;
- presença de atividade no clock, entradas e saídas;
- sinais constantes ou sem eventos relevantes.

Um SAIF vazio, com duração zero, sem alternâncias na lógica do DUT ou gerado a
partir de uma simulação incompleta deverá ser rejeitado.

### 11.10 Etapa 70 — anotação e potência

O SAIF deverá ser lido sobre o mesmo checkpoint e a mesma revisão de RTL que
originaram a simulação. O pipeline recusará combinações cujos hashes não
coincidam.

O `strip_path` será derivado do wrapper e do nome da instância DUT. Não poderá
ser fixado como `apatb_snn_to_hls_top/AESL_inst_snn_to_hls`.

Após `read_saif`, o ambiente verificará:

- quantidade e percentual de nets correspondentes;
- nets não correspondentes;
- clocks reconhecidos;
- cobertura de atividade nos principais blocos hierárquicos;
- nível de confiança reportado pelo Vivado.

Por padrão, cobertura inferior a `saif_min_match_percent` falhará. O usuário
poderá reduzir o limite explicitamente, e essa exceção aparecerá no resumo.
O parser deverá persistir potência, confiança, contagens de nets e marcadores
de qualidade antes de aplicar essa política. Desse modo, uma reprovação
preservará o diagnóstico em `power_summary.json` e nos resumos, mas os valores
serão marcados como provisórios e a etapa continuará em `failed`.

Relatórios mínimos:

- potência total, dinâmica e estática;
- potência por recurso e por hierarquia;
- clocks e taxas de atividade;
- condições ambientais e de processo;
- confiança da estimativa;
- relatório de nets não anotadas;
- arquivo RPX ou equivalente, quando suportado;
- versão textual e versão estruturada para comparação automática.

### 11.11 Etapa 80 — relatório consolidado

Cada execução produzirá:

- `reports/summary.md`, para leitura humana;
- `reports/summary.json`, para automação;
- `reports/artifacts.json`, com caminhos e hashes;
- `reports/reproduce.sh` ou lista de comandos equivalente;
- links para logs e relatórios originais das ferramentas.

O resumo conterá pelo menos:

| Grupo | Métricas |
|---|---|
| Funcional | amostras, acertos, acurácia, mismatches |
| Carga | amostras executadas, batches, steps/amostra, steps lógicos totais |
| Desempenho | duração SAIF, latência média amortizada por step e por amostra |
| HLS | clock, latência, II, BRAM, DSP, FF, LUT, URAM |
| Vivado | utilização, WNS/TNS, warnings críticos |
| Atividade | duração, ciclos, amostras, match SAIF |
| Potência | total, dinâmica, estática, confiança |
| Energia | captura total, média por step e média por amostra |
| Proveniência | hashes, versões, commit, perfil |

O `summary.md` deverá separar explicitamente a estimativa HLS da utilização
pós-síntese do Vivado OOC. A seção HLS não deve ser apresentada como uso
pós-síntese. A seção Vivado deve deixar claro que a síntese é out-of-context,
que o percentual usa a capacidade total do FPGA, que recursos reservados por
plataforma ou shell não são descontados e que BRAM pode aparecer fracionário
porque uma RAMB18 ocupa meio tile.

As métricas de desempenho e energia serão calculadas sobre a janela SAIF
completa:

```text
steps_totais     = amostras_executadas × steps_por_amostra
latência/step    = duração_SAIF / steps_totais
latência/amostra = duração_SAIF / amostras_executadas
energia          = potência_média × duração_SAIF
```

Essas latências são amortizadas e incluem overhead e intervalos do testbench;
não substituem uma medição isolada do protocolo `ap_start`–`ap_done`. Steps
lógicos também não serão confundidos com transações do DUT em backends
event-driven.

## 12. Perfil `power-accurate`

Esse perfil estenderá o pipeline após a síntese:

1. executar opt, place e route;
2. validar timing ou registrar claramente sua violação;
3. gravar checkpoint implementado;
4. gerar netlist e SDF pós-route;
5. executar simulação temporal;
6. capturar novo SAIF na mesma janela funcional;
7. ler o SAIF no checkpoint implementado;
8. gerar o relatório final de potência.

O ambiente preservará ambos os resultados, permitindo comparar
`post-synth` e `power-accurate`. O relatório não deverá substituir um pelo
outro silenciosamente.

## 13. Cache e retomada

Cada etapa terá um hash de entrada e um arquivo de status. Uma etapa poderá ser
reutilizada somente se:

- terminou com sucesso;
- seus artefatos ainda existem e mantêm os hashes;
- todas as entradas e configurações relevantes são idênticas;
- a versão das ferramentas é compatível.

Exemplos de invalidação:

- mudar `tb_data` ou o testbench invalida CSim, simulação e SAIF, mas pode
  preservar sínteses quando os fontes do componente permanecerem idênticos;
- mudar peso, quantização, backend ou qualquer fonte invalida a síntese HLS e
  todas as etapas seguintes;
- mudar clock ou part invalida síntese HLS, Vivado e potência;
- mudar o relatório global de utilização ou a normalização do resumo invalida
  a reutilização da etapa `vivado-synth`; caches antigos sem
  `utilization_device_post_synth.rpt` e `reports/utilization_summary.json`
  devem ser considerados obsoletos;
- mudar somente temperatura invalida o relatório de potência, não a simulação;
- mudar a janela de atividade invalida simulação, SAIF e potência.

Comandos previstos:

```bash
python -m sim status --run <run-id>
python -m sim resume --run <run-id>
python -m sim run --project <pasta> --from vivado-synth
python -m sim run --project <pasta> --to post-synth-sim
```

## 14. Observabilidade e diagnóstico

- Cada processo externo terá um log próprio e saída acompanhável no terminal.
- O log iniciará com o comando resolvido, diretório e variáveis relevantes,
  ocultando eventuais segredos.
- Warnings críticos serão extraídos para o resumo, sem apagar o log original.
- O status será atualizado atomicamente para sobreviver a interrupções.
- `Ctrl+C` encerrará processos filhos e marcará a etapa como interrompida.
- O ambiente não apagará pastas de origem nem artefatos de outra execução.

## 15. Integração com a API NeuroHLS existente

O ambiente deverá orquestrar as operações de execução existentes sem assumir a
responsabilidade de gerar a rede. As operações atuais que servem de base são:

- `run_csim`;
- `run_synth`;
- `run_cosim(..., setup_only=True)`;
- `run_export_design`;
- `run_vivado_synthesis`;
- `generate_post_synth_saif`;
- `generate_power_report`;
- `run_power_report_with_saif`.

Antes de serem usadas como infraestrutura genérica, essas operações precisarão
aceitar de forma consistente:

- part;
- frequência ou período de clock;
- top;
- nome de projeto e solução;
- diretórios de entrada e saída;
- nomes do wrapper e da instância DUT;
- scope e `strip_path` SAIF;
- quantidade de jobs e timeouts;
- política de erro.

O orquestrador não deverá depender de métodos que capturam uma exceção, apenas
imprimem uma mensagem e retornam sucesso aparente.

## 16. Lacunas identificadas no fluxo atual

O protótipo existente demonstra que o fluxo é viável, mas contém pontos que
precisam ser resolvidos:

1. uso fixo do part `xcu250-figd2104-2L-e` em etapas Vivado;
2. nomes fixos `snn_to_hls`, `apatb_snn_to_hls_top` e
   `AESL_inst_snn_to_hls`;
3. frequências diferentes entre síntese e relatório de potência em alguns
   scripts;
4. dependência da estrutura interna e do autotestbench de uma versão do HLS;
5. alteração textual de arquivos gerados para remover monitores;
6. caminhos Windows específicos para Vivado 2025.2;
7. ausência de isolamento e manifesto por execução;
8. ausência de um gate obrigatório para percentual de nets anotadas;
9. tratamento inconsistente de códigos de saída;
10. relatórios espalhados dentro da solução Vitis.

O novo ambiente deverá eliminar essas lacunas, e não apenas copiar os scripts
específicos de `event_driven_scnn`.

## 17. Requisitos funcionais

### RF-01 — seleção de componente

O usuário deverá selecionar qualquer componente válido por um único argumento,
sem editar Python, TCL, Verilog ou caminhos internos.

### RF-02 — execução completa

O comando `run --profile post-synth` deverá executar todas as etapas necessárias
e retornar zero somente quando validação, SAIF e relatório forem válidos.

### RF-03 — execução parcial

O usuário deverá poder limitar, repetir ou retomar etapas sem reconstruir
artefatos ainda válidos.

### RF-04 — parametrização

Part, clock, top, solução, ferramentas, jobs, janela de atividade e condições
de potência deverão vir da configuração ou ser descobertos no projeto.

### RF-05 — validação funcional

A saída pós-síntese deverá ser comparada à referência para todas as amostras
executadas. A tolerância será configurável apenas para saídas não binárias.

### RF-06 — atividade

O ambiente deverá gerar SAIF não vazio, com duração conhecida, atividade
representativa e scope limitado ao DUT.

### RF-07 — rastreabilidade

Todo relatório deverá ser rastreável até o hash do projeto NeuroHLS, fontes,
testbench, dados de teste, RTL, checkpoint, SAIF, configuração e versões
exatas.

### RF-08 — comparação

O formato estruturado deverá permitir comparar execuções de duas redes ou dois
perfis sem analisar manualmente os relatórios do Vivado.

### RF-09 — portabilidade

Linux será a plataforma primária. O design não deverá introduzir caminhos
absolutos de uma instalação específica. Suporte a Windows poderá ser
adicionado por um adaptador de ambiente, preservando a mesma estrutura de
projeto.

## 18. Requisitos não funcionais

- **determinismo:** mesmas entradas e versões devem produzir configuração e
  comandos equivalentes;
- **auditabilidade:** nenhum artefato utilizado será sobrescrito sem registro;
- **idempotência:** retomar uma etapa concluída não deverá corromper a run;
- **isolamento:** duas execuções não compartilharão projetos graváveis;
- **diagnóstico:** erros deverão indicar etapa, comando, log e artefato;
- **extensibilidade:** novos simuladores ou perfis poderão implementar a mesma
  interface de estágio;
- **testabilidade:** schemas, hashes, parsing de relatórios e construção de
  comandos terão testes unitários sem exigir licença AMD;
- **segurança:** caminhos fornecidos pelo YAML serão normalizados e não poderão
  causar remoção fora do diretório da execução.

## 19. Critérios de aceite do MVP

O MVP será aceito quando:

1. duas pastas de projeto NeuroHLS com shapes distintos puderem ser executadas
   apenas trocando o valor de `--project`;
2. nenhum TCL específico das duas redes for necessário;
3. o fluxo completo produzir CSim, síntese HLS, exportação, síntese Vivado,
   simulação pós-síntese, SAIF e relatório de potência;
4. uma divergência pós-síntese interromper o fluxo;
5. SAIF vazio ou com match inferior ao limite interromper o fluxo;
6. part e clock do relatório coincidirem com os da síntese;
7. `summary.md` e `summary.json` identificarem todos os artefatos, versões,
   carga, latências, potências, energias e a validade condicionada à cobertura;
8. uma execução interrompida puder ser retomada;
9. mudar somente o componente não exigir limpeza manual;
10. testes automatizados cobrirem validação de config, construção dos comandos,
    cache, parsing do SAIF/power e condições de falha.

Como validação inicial, recomenda-se usar:

- uma rede recorrente event-driven;
- a `event_driven_scnn`;
- uma rede time-driven simples.

Isso exercita recorrência, convolução e os dois modelos de execução.

## 20. Plano de implementação

### P0 — fundação

- concluída: CLI, configuração YAML, criação de runs, hashes, logs e preflight;
- concluída: validação e cópia isolada de projetos NeuroHLS;
- concluída: testes unitários sem ferramentas AMD;
- pendente: schemas JSON formais, se forem necessários além da validação Python.

### P1 — fluxo pós-síntese

- implementada: integração CSim/síntese/exportação;
- implementado: TCL genérico de síntese OOC;
- implementada: simulação funcional pós-síntese com autotestbench HLS;
- implementadas: captura, validação e anotação SAIF;
- implementado: resumo consolidado com carga, latência, potência, energia e
  marcadores de validade;
- pendente: validação de integração completa em projetos representativos.

### P2 — troca transparente de redes

- descoberta de interface;
- wrapper/testbench gerado;
- suporte validado a shapes e tops diferentes.

### P3 — potência de maior confiança

- implementação completa;
- netlist e SDF pós-route;
- simulação temporal;
- relatório no checkpoint implementado;
- comparação pós-síntese versus pós-implementação.

### P4 — comparação e automação

- comando de comparação;
- execução em lote;
- limites de regressão de acurácia, recursos, timing e potência;
- integração opcional com CI em máquinas licenciadas.

## 21. Decisões pendentes

Antes da implementação, deverão ser confirmados:

- FPGA/placa padrão do laboratório;
- versão AMD que será oficialmente homologada;
- projetos NeuroHLS usados como smoke test;
- limite mínimo de anotação SAIF;
- se potência pós-route entra no MVP ou permanece em P3;
- formato de referência para saídas não binárias;
- política para armazenar ou descartar artefatos grandes.

Essas decisões alteram defaults e critérios de aceite, mas não a arquitetura
implementada.
