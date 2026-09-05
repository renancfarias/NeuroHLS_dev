"""Renderização de TCL parametrizado para Vitis HLS e Vivado."""

from pathlib import Path
from typing import Iterable, Optional, Sequence

from .config import RunConfig
from .utils import tcl_quote


def vitis_wrapper(source_tcl: Path, arguments: Sequence[object]) -> str:
    """Cria um wrapper Vitis que preserva os argumentos esperados pelo projeto."""
    rendered_arguments = " ".join(tcl_quote(argument) for argument in arguments)
    return "\n".join(
        (
            "# Gerado pelo ambiente sim; não editar.",
            "set argv [list {}]".format(rendered_arguments),
            "set argc {}".format(len(arguments)),
            "source {}".format(tcl_quote(source_tcl.resolve())),
            "",
        )
    )


def cosim_setup_tcl(project_name: str, solution_name: str) -> str:
    return "\n".join(
        (
            "# Gerado pelo ambiente sim; configura vetores sem executar RTL.",
            "open_project {}".format(tcl_quote(project_name)),
            "open_solution {}".format(tcl_quote(solution_name)),
            "cosim_design -setup",
            "exit",
            "",
        )
    )


def cosim_tcl(project_name: str, solution_name: str) -> str:
    """Executa a co-simulação RTL e mede latência real, com stalls incluídos.

    Diferente da simulação pós-síntese, que roda o netlist e custa horas, esta
    roda o RTL comportamental e é a fonte de latência para designs cuja
    latência a síntese reporta como ``undef`` -- em dataflow com FIFOs finitos,
    o tempo depende da contrapressão, que nenhum relatório estático captura.
    """
    return "\n".join(
        (
            "# Gerado pelo ambiente sim; co-simulação RTL sobre os vetores do setup.",
            "open_project {}".format(tcl_quote(project_name)),
            "open_solution {}".format(tcl_quote(solution_name)),
            "cosim_design -rtl verilog -tool xsim",
            "exit",
            "",
        )
    )


def export_tcl(project_name: str, solution_name: str) -> str:
    return "\n".join(
        (
            "# Gerado pelo ambiente sim; exporta o componente HLS para Vivado.",
            "open_project {}".format(tcl_quote(project_name)),
            "open_solution {}".format(tcl_quote(solution_name)),
            "export_design -format ip_catalog",
            "exit",
            "",
        )
    )


def clock_override_xdc(config: RunConfig) -> str:
    """Renderiza o clock da plataforma como constraint aplicada na síntese."""
    return """# Gerado pelo ambiente sim; sobrescreve o clock exportado pelo HLS.
create_clock -name {clock_name} -period {clock_period:.12g} [get_ports {clock_name}]
set_clock_uncertainty {clock_uncertainty:.12g} [get_clocks {clock_name}]
""".format(
        clock_name=tcl_quote(config.clock_name),
        clock_period=config.clock_period_ns,
        clock_uncertainty=config.clock_uncertainty_ns,
    )


def vivado_synth_tcl(
    config: RunConfig,
    top: str,
    rtl_dir: Path,
    xdc_files: Iterable[Path],
    dcp_path: Path,
    netlist_path: Path,
    utilization_path: Path,
    device_utilization_path: Path,
    timing_path: Path,
    methodology_path: Path,
) -> str:
    """Renderiza síntese OOC sem part/top/hierarquia fixos."""
    rendered_xdc = "\n".join(
        "read_xdc {}".format(tcl_quote(Path(path).resolve())) for path in xdc_files
    )
    return """# Gerado pelo ambiente sim; síntese Vivado OOC.
set part {part}
set top {top}
set rtl_dir {rtl_dir}
set clock_name {clock_name}
set clock_period {clock_period:.12g}

proc collect_hdl {{dir_path}} {{
    set result {{}}
    foreach item [glob -nocomplain -directory $dir_path *] {{
        if {{[file isdirectory $item]}} {{
            foreach nested [collect_hdl $item] {{ lappend result $nested }}
        }} else {{
            set extension [string tolower [file extension $item]]
            if {{$extension eq ".v" || $extension eq ".sv"}} {{ lappend result $item }}
        }}
    }}
    return $result
}}

create_project -in_memory -part $part
set hdl_files [collect_hdl $rtl_dir]
if {{![llength $hdl_files]}} {{ error "Nenhum HDL exportado foi encontrado em $rtl_dir" }}
set include_dirs [list $rtl_dir]
foreach hdl_file $hdl_files {{
    lappend include_dirs [file dirname $hdl_file]
}}
set include_dirs [lsort -unique $include_dirs]
set_property INCLUDE_DIRS $include_dirs [current_fileset]
read_verilog -sv $hdl_files
{rendered_xdc}
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy none -keep_equivalent_registers
set clock_port [get_ports -quiet $clock_name]
if {{![llength $clock_port]}} {{
    error "A porta de clock configurada não existe no top sintetizado: $clock_name"
}}
set configured_clocks [get_clocks -quiet -of_objects $clock_port]
if {{![llength $configured_clocks]}} {{
    error "Nenhum clock foi aplicado à porta configurada: $clock_name"
}}
set actual_clock_period [get_property PERIOD [lindex $configured_clocks 0]]
if {{abs($actual_clock_period - $clock_period) > 0.001}} {{
    error "Período do clock $clock_name diverge da configuração: $actual_clock_period ns != $clock_period ns"
}}
report_utilization -hierarchical -file {utilization_path}
report_utilization -file {device_utilization_path}
report_timing_summary -file {timing_path}
report_methodology -file {methodology_path}
write_verilog -force -mode funcsim {netlist_path}
write_checkpoint -force {dcp_path}
close_project
exit
""".format(
        part=tcl_quote(config.part),
        top=tcl_quote(top),
        rtl_dir=tcl_quote(rtl_dir.resolve()),
        clock_name=tcl_quote(config.clock_name),
        clock_period=config.clock_period_ns,
        rendered_xdc=rendered_xdc,
        utilization_path=tcl_quote(utilization_path.resolve()),
        device_utilization_path=tcl_quote(device_utilization_path.resolve()),
        timing_path=tcl_quote(timing_path.resolve()),
        methodology_path=tcl_quote(methodology_path.resolve()),
        netlist_path=tcl_quote(netlist_path.resolve()),
        dcp_path=tcl_quote(dcp_path.resolve()),
    )


def xsim_saif_tcl(saif_path: Path, autotb_top: str, dut_instance: str) -> str:
    capture_scope = "/{}/{}".format(autotb_top, dut_instance)
    return "\n".join(
        (
            "# Gerado pelo ambiente sim; captura atividade da simulação pós-síntese.",
            "load_features simulator",
            "set capture_scope {}".format(tcl_quote(capture_scope)),
            "set capture_scopes [get_scopes -quiet $capture_scope]",
            "if {[llength $capture_scopes] != 1} {",
            '    error "Escopo DUT inválido para captura SAIF: $capture_scope"',
            "}",
            "set previous_scope [current_scope]",
            "current_scope [lindex $capture_scopes 0]",
            "set saif_objects [get_objects -r *]",
            "if {![llength $saif_objects]} {",
            "    current_scope $previous_scope",
            '    error "Nenhum sinal encontrado no escopo de captura SAIF: $capture_scope"',
            "}",
            "open_saif {}".format(tcl_quote(saif_path.resolve())),
            "log_saif $saif_objects",
            "current_scope $previous_scope",
            "run all",
            "close_saif",
            "quit",
            "",
        )
    )


def power_tcl(
    config: RunConfig,
    dcp_path: Path,
    saif_path: Optional[Path],
    strip_path: Optional[str],
    report_path: Path,
    unmatched_path: Optional[Path],
) -> str:
    """Emite o relatório de potência sobre o checkpoint pós-síntese.

    Com ``saif_path``, a atividade medida na simulação pós-síntese é anotada no
    checkpoint.  Sem ele, a estimativa é vectorless: as taxas de transição
    padrão declaradas na configuração valem para todo o design.  As duas formas
    validam o clock do checkpoint da mesma maneira, porque um período divergente
    invalidaria a potência dinâmica em qualquer um dos dois modos.
    """
    if saif_path is None:
        activity_block = """set_switching_activity -default_toggle_rate {toggle_rate:.12g} \\
    -default_static_probability {static_probability:.12g}""".format(
            toggle_rate=config.default_toggle_rate_percent,
            static_probability=config.default_static_probability,
        )
        header = (
            "# Gerado pelo ambiente sim; potência vectorless "
            "(taxas de transição padrão, sem SAIF)."
        )
    else:
        if strip_path is None or unmatched_path is None:
            raise ValueError(
                "A anotação SAIF exige strip_path e unmatched_path"
            )
        activity_block = "read_saif -strip_path {strip_path} -out_file {unmatched_path} {saif_path}".format(
            strip_path=tcl_quote(strip_path),
            unmatched_path=tcl_quote(Path(unmatched_path).resolve()),
            saif_path=tcl_quote(Path(saif_path).resolve()),
        )
        header = "# Gerado pelo ambiente sim; potência baseada em SAIF pós-síntese."
    return """{header}
create_project -in_memory -part {part}
open_checkpoint {dcp_path}
set clock_name {clock_name}
set clock_period {clock_period:.12g}
set clock_port [get_ports -quiet $clock_name]
if {{![llength $clock_port]}} {{
    error "A porta de clock configurada não existe no checkpoint: $clock_name"
}}
set configured_clocks [get_clocks -quiet -of_objects $clock_port]
if {{![llength $configured_clocks]}} {{
    error "Nenhum clock foi preservado no checkpoint para a porta: $clock_name"
}}
set actual_clock_period [get_property PERIOD [lindex $configured_clocks 0]]
if {{abs($actual_clock_period - $clock_period) > 0.001}} {{
    error "Período do clock $clock_name diverge da configuração: $actual_clock_period ns != $clock_period ns"
}}
{activity_block}
report_power -file {report_path}
close_project
exit
""".format(
        header=header,
        part=tcl_quote(config.part),
        dcp_path=tcl_quote(dcp_path.resolve()),
        clock_name=tcl_quote(config.clock_name),
        clock_period=config.clock_period_ns,
        activity_block=activity_block,
        report_path=tcl_quote(report_path.resolve()),
    )
