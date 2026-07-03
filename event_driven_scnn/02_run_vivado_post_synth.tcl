set ROOT [file normalize [file dirname [info script]]]
set COMPONENT_DIR [file join $ROOT event_driven_scnn]
set HLS_DIR [file join $COMPONENT_DIR hls]
set RTL_DIR [file join $HLS_DIR impl verilog]
set SIM_DIR [file join $HLS_DIR sim verilog]
set VIVADO_WORK [file join $HLS_DIR impl vivado_power_work]
set VIVADO_PROJECT vivado_power
set POWER_REPORT [file join $HLS_DIR reports power_report.txt]

set SIM_ONLY 0
foreach arg $argv {
    if {$arg eq "sim_only" || $arg eq "-sim_only" || $arg eq "--sim-only"} {
        set SIM_ONLY 1
    }
}
if {[info exists ::env(SIM_ONLY)] && $::env(SIM_ONLY) ne "" && $::env(SIM_ONLY) ne "0"} {
    set SIM_ONLY 1
}

set VIVADO_XPR [file join $VIVADO_WORK ${VIVADO_PROJECT}.xpr]
set SYNTH_DCP [file join $VIVADO_WORK ${VIVADO_PROJECT}.runs synth_1 scnn.dcp]

if {$SIM_ONLY} {
    if {![file exists $VIVADO_XPR]} {
        error "Missing Vivado project for sim_only mode: $VIVADO_XPR"
    }
    if {![file exists $SYNTH_DCP]} {
        error "Missing synthesized checkpoint for sim_only mode: $SYNTH_DCP"
    }

    puts "SIM_ONLY mode: reusing synthesized run from $VIVADO_XPR"
    open_project $VIVADO_XPR
    open_run synth_1
} else {
    create_project -force $VIVADO_PROJECT $VIVADO_WORK -part xcu250-figd2104-2L-e

    set verilog_files [glob -nocomplain -directory $RTL_DIR *.v]
    add_files -norecurse $verilog_files
    set_property top scnn [current_fileset]
    update_compile_order -fileset sources_1

    # limit threads to prevent memory/cpu exhaustion
    set_param synth.maxThreads 1
    set_param general.maxThreads 1

    # timing constraint
    set xdc_file [file join $VIVADO_WORK clock.xdc]
    file mkdir $VIVADO_WORK
    set xdc_fd [open $xdc_file w]
    puts $xdc_fd "create_clock -name ap_clk -period 20.0 \[get_ports ap_clk\]"
    close $xdc_fd
    add_files -fileset constrs_1 -norecurse $xdc_file

    # Make synthesis lighter without editing the generated run script.
    set synth_run [get_runs synth_1]
    set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE RuntimeOptimized $synth_run

    set SYNTH_RUN_SCRIPT [file join $VIVADO_WORK $VIVADO_PROJECT.runs synth_1 scnn.tcl]
    launch_runs synth_1 -scripts_only

    if {[file exists $SYNTH_RUN_SCRIPT]} {
        set synth_fd [open $SYNTH_RUN_SCRIPT r]
        set synth_script [read $synth_fd]
        close $synth_fd

        set old_cmd {synth_design -top scnn -part xcu250-figd2104-2L-e}
        set new_cmd {synth_design -flatten_hierarchy none -top scnn -part xcu250-figd2104-2L-e -directive RuntimeOptimized}
        if {[string match "*synth_design -top scnn -part xcu250-figd2104-2L-e*" $synth_script]} {
            regsub -all -- $old_cmd $synth_script $new_cmd synth_script
            set synth_fd [open $SYNTH_RUN_SCRIPT w]
            puts -nonewline $synth_fd $synth_script
            close $synth_fd
            puts "Patched synth run script with out_of_context mode: $SYNTH_RUN_SCRIPT"
        } else {
            puts "Warning: synth_design command not found in $SYNTH_RUN_SCRIPT"
        }
    } else {
        puts "Warning: synth run script not found yet: $SYNTH_RUN_SCRIPT"
    }

    reset_run synth_1
    launch_runs synth_1 -jobs 1
    wait_on_run synth_1

    open_run synth_1
}

# Simulation setup
proc collect_sim_files {dir_path} {
    set result {}
    foreach item [glob -nocomplain -directory $dir_path *] {
        if {[file isdirectory $item]} {
            foreach nested [collect_sim_files $item] {
                lappend result $nested
            }
        } else {
            set ext [string tolower [file extension $item]]
            if {$ext in {".v" ".sv" ".vh" ".svh" ".dat" ".txt"}} {
                lappend result $item
            }
        }
    }
    return $result
}

proc collect_prj_sources {prj_path base_dir} {
    set result {}
    set fd [open $prj_path r]
    while {[gets $fd line] >= 0} {
        if {[regexp {^[[:space:]]*(v|sv)[[:space:]]+[^[:space:]]+[[:space:]]+"([^"]+)"} $line -> _ rel_path]} {
            lappend result [file normalize [file join $base_dir $rel_path]]
        }
    }
    close $fd
    return $result
}

proc is_post_synth_monitor_file {path} {
    set name [file tail $path]
    return [expr {
        $name eq "AESL_deadlock_detection_unit.v" ||
        $name eq "AESL_deadlock_report_unit.v" ||
        $name eq "AESL_deadlock_detector.v" ||
        $name eq "dataflow_monitor.sv"
    }]
}

foreach f [get_files -quiet -of_objects [get_filesets sim_1]] {
    if {[is_post_synth_monitor_file $f] || [file tail $f] eq "post_synth_monitor_stubs.sv"} {
        remove_files -quiet $f
    }
}

set sim_files [collect_prj_sources [file join $SIM_DIR scnn.prj] $SIM_DIR]
set valid_sim_files {}
foreach f $sim_files {
    if {![string match "*_dataflow_ana.wcfg" $f] && ![is_post_synth_monitor_file $f] && [file exists $f]} {
        lappend valid_sim_files $f
    }
}
if {[llength $valid_sim_files] > 0} {
    add_files -fileset sim_1 -norecurse $valid_sim_files
}

set POST_SYN_STUBS [file join $VIVADO_WORK post_synth_monitor_stubs.sv]
set stub_fd [open $POST_SYN_STUBS w]
puts $stub_fd {`timescale 1ns/1ps

module AESL_deadlock_detector(
    input dl_reset,
    input all_finish,
    input dl_clock);
endmodule

module dataflow_monitor(
    input clock,
    input reset,
    input finish);
endmodule
}
close $stub_fd
add_files -fileset sim_1 -norecurse $POST_SYN_STUBS

set_property top apatb_scnn_top [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
update_compile_order -fileset sim_1

# The HLS-generated autotb wraps RTL-only dependence checks in `ifndef POST_SYN`.
# Post-synthesis simulation must define it because those checks and HLS
# monitors reference internal signals that synthesis can optimize/rename away.
set_property verilog_define {POST_SYN} [get_filesets sim_1]
set_property -name {xsim.compile.xvlog.more_options} -value {-d POST_SYN} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif_all_signals} -value {true} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif_scope} -value {/apatb_scnn_top/AESL_inst_scnn} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif} -value {scnn_post_synth.saif} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {all} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.xsim.more_options} -value {-testplusarg UVM_VERBOSITY=UVM_NONE -testplusarg UVM_TESTNAME=scnn_test_lib -testplusarg UVM_TIMEOUT=20000000000000 -testplusarg UVM_NO_RELNOTES} -objects [get_filesets sim_1]

# Copy init txt data to sim run directory so xsim finds them
set TARGET_SIM_DIR [file join $VIVADO_WORK $VIVADO_PROJECT.sim sim_1 synth func xsim]
set TARGET_TV_DIR [file normalize [file join $TARGET_SIM_DIR .. tv]]
set SOURCE_TV_DIR [file join $HLS_DIR sim tv]
set SAIF_FILE [file join $TARGET_SIM_DIR scnn_post_synth.saif]
set COPIED_SAIF_FILE [file join $SIM_DIR scnn_post_synth.saif]
file mkdir $TARGET_SIM_DIR
file delete -force $SAIF_FILE $COPIED_SAIF_FILE
if {[file exists $SOURCE_TV_DIR]} {
    file delete -force $TARGET_TV_DIR
    file copy -force $SOURCE_TV_DIR $TARGET_TV_DIR
} else {
    error "Missing HLS test-vector directory: $SOURCE_TV_DIR"
}
foreach f [collect_sim_files $SIM_DIR] {
    if {[string match "*.txt" $f] || [string match "*.dat" $f]} {
        file copy -force $f $TARGET_SIM_DIR
    }
}

launch_simulation -mode post-synthesis -type functional

set SIM_LOG [file join $TARGET_SIM_DIR simulate.log]
if {[file exists $SIM_LOG]} {
    set sim_log_fd [open $SIM_LOG r]
    set sim_log_text [read $sim_log_fd]
    close $sim_log_fd
    if {[string match "*UVM_FATAL*" $sim_log_text] || [string match "*ERROR: Simulation using HLS TB failed*" $sim_log_text]} {
        error "Post-synthesis simulation failed. Check $SIM_LOG"
    }
}

if {![file exists $SAIF_FILE]} {
    error "SAIF file not generated at $SAIF_FILE"
} else {
    puts "SAIF file generated successfully at $SAIF_FILE"
    file copy -force $SAIF_FILE $COPIED_SAIF_FILE
    puts "SAIF file copied to $COPIED_SAIF_FILE"
}

close_project
quit
