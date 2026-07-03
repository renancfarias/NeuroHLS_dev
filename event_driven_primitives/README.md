# Event-driven NIR primitives

Esta pasta consolida as primitivas NIR encontradas nas pastas `event_driven*`.

As implementacoes conflitantes foram resolvidas usando as versoes de
`event_driven_scnn`, conforme solicitado:

- `affine_layer`
- `if_layer`
- `linear_layer`
- `sumpool2d_layer`

As funcoes que usavam o formato legado de `spike_t` (`index` e `last_feature`)
foram convertidas para o formato novo (`type`, `channel_idx`, `height_idx` e
`width_idx`).

As antigas subpastas `scnn` e `srnn` foram mescladas no diretorio raiz. O
arquivo `types.h` unico usa a definicao de tipos mais ampla de `srnn`, para
preservar precisao nas primitivas recorrentes.

- `affine.h`: contem `affine_layer` e `Affine`.
- `linear.h`: contem `linear_layer`, `linear_layer_pruning` e `Linear`.
- `conv2d.h`: contem `conv2d_layer_no_bias` e `conv2d_layer`.
- `lif.h` e `lif_utils.*`: contem a `lif_layer` e auxiliares LIF convertidos
  de `event_driven`.
- Definicoes duplicadas em `event_driven_srnn_subtract` eram identicas e foram
  omitidas.

Funcoes legadas ja substituidas por versoes `scnn`, como o `if_layer` antigo,
foram omitidas.

Arquivos de rede/top-level, testbench, debug e artefatos HLS nao foram copiados.
