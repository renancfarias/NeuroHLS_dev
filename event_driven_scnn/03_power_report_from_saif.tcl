set ROOT [file normalize [file dirname [info script]]]
set COMPONENT_DIR [file join $ROOT event_driven_scnn]
set HLS_DIR [file join $COMPONENT_DIR hls]
set VIVADO_WORK [file join $HLS_DIR impl vivado_power_work]
set VIVADO_PROJECT vivado_power
set SAIF_FILE [file join $HLS_DIR sim verilog scnn_post_synth.saif]
set SYNTH_DCP [file join $VIVADO_WORK $VIVADO_PROJECT.runs synth_1 scnn.dcp]
set POWER_REPORT [file join $HLS_DIR reports power_report.txt]

if {![file exists $SAIF_FILE]} {
	error "Missing SAIF activity file: $SAIF_FILE"
}

if {![file exists $SYNTH_DCP]} {
	error "Missing synthesized checkpoint: $SYNTH_DCP"
}

open_checkpoint $SYNTH_DCP

create_clock -name ap_clk -period 6.667 [get_ports ap_clk]
read_saif $SAIF_FILE -strip_path apatb_scnn_top/AESL_inst_scnn

file mkdir [file dirname $POWER_REPORT]
report_power -file $POWER_REPORT
puts "Power report generated successfully at $POWER_REPORT"

close_project

exit