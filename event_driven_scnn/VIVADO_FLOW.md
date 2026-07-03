# Flow de exportação, SAIF e power

Ordem sugerida:

1. Gerar a exportação do componente para Vivado:
   - `01_export_component_to_vivado.tcl`
2. Rodar a simulação pós-síntese e gerar o SAIF:
   - `02_post_synth_saif.tcl`
3. Gerar o relatório de power usando o SAIF:
   - `03_power_report_from_saif.tcl`

Arquivos principais gerados:

- `event_driven_scnn/hls/impl/export.zip`
- `event_driven_scnn/hls/sim/verilog/scnn_post_synth.saif`
- `event_driven_scnn/hls/reports/power_report.txt`