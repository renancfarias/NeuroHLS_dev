import json
import os
import queue
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from sim.config import load_config
from sim.errors import CommandError, StageError, ValidationError
from sim.pipeline import (
    Pipeline,
    derive_summary_metrics,
    parse_csim_accuracy,
    parse_power_report,
    parse_saif,
    parse_testbench_workload,
    parse_vivado_utilization_report,
    saif_duration_seconds,
)
from sim.project import validate_project
from sim.tcl import clock_override_xdc, power_tcl, vivado_synth_tcl, xsim_saif_tcl
from sim.utils import CommandRunner, _terminate_process_group


def create_project(root: Path, top: str = "my_top") -> Path:
    project = root / "generated_project"
    (project / "neuro_hls_functions").mkdir(parents=True)
    (project / "tb_data").mkdir()
    files = {
        "0_create_project.tcl": "open_project -reset vitis_proj\nexit\n",
        "1_csim.tcl": "exit\n",
        "2_synth.tcl": "set_top {}\ncsynth_design\nexit\n".format(top),
        "3_cosim.tcl": "exit\n",
        "snn_implementation.cpp": "#include \"snn_implementation.h\"\n",
        "snn_implementation.h": "void {}();\n".format(top),
        "testbench.cpp": "int main() { return 0; }\n",
        "neuron_params.h": "#pragma once\n",
        "quantization.h": "#pragma once\n",
        "neuro_hls_functions/bit_type.h": "#pragma once\n",
        "tb_data/data.txt": "0\n",
        "tb_data/targets.txt": "0\n",
    }
    for relative, contents in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return project


def utilization_report_text(
    part: str = "xcu250-figd2104-2L-e",
    design_state: str = "Synthesized",
    lut_label: str = "CLB LUTs",
    ff_label: str = "CLB Registers",
    include_uram: bool = True,
    include_prohibited: bool = True,
) -> str:
    rows = [
        (lut_label, "4945", "0", "1728000", "0.2862"),
        (ff_label, "5768", "0", "3456000", "0.1669"),
        ("Block RAM Tile", "30.5", "0", "2688", "1.1347"),
        ("DSPs", "89", "0", "12288", "0.7243"),
    ]
    if include_uram:
        rows.append(("URAM", "0", "0", "1280", "0.0000"))
    if include_prohibited:
        table_header = "| Site Type            | Used     | Fixed | Prohibited | Available | Util% |"
        table_rule = "+----------------------+----------+-------+------------+-----------+-------+"
        table_rows = "\n".join(
            "| {0:<18} | {1:>7} | {2:>5} | {2:>10} | {3:>10} | {4:>7} |".format(*row)
            for row in rows
        )
    else:
        table_header = "| Site Type            | Used     | Fixed | Available  | Util% |"
        table_rule = "+----------------------+----------+-------+------------+--------+"
        table_rows = "\n".join(
            "| {0:<18} | {1:>7} | {2:>5} | {3:>10} | {4:>7} |".format(*row)
            for row in rows
        )
    return """Copyright 1986-2022 Xilinx, Inc. All Rights Reserved. Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Tool Version : Vivado v.2025.2 (lin64) Build 6299465 Fri Nov 14 12:34:56 MST 2025
| Date         : Sun Jul 26 01:34:25 2026
| Host         : localhost.localdomain running 64-bit AlmaLinux 9.6 (Sage Margay)
| Command      : report_utilization -file /tmp/utilization_device_post_synth.rpt
| Design       : snn_to_hls
| Device       : {part}
| Speed File   : -2L
| Design State : {design_state}
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Utilization Design Information

Table of Contents
-----------------
1. Device Utilization Summary

1. Device Utilization Summary
-----------------------------

{table_rule}
{table_header}
{table_rule}
{table_rows}
{table_rule}
""".format(
        part=part,
        design_state=design_state,
        table_header=table_header,
        table_rule=table_rule,
        table_rows=table_rows,
    )


def power_report_text(
    match_percent: str = "52",
    matched_nets: int = 13439,
    total_nets: int = 25972,
    dynamic_power: str = "0.103",
) -> str:
    return """Power Report
1. Summary
+--------------------------+-----------------------+
| Total On-Chip Power (W)  | 3.049                 |
| Dynamic (W)              | {dynamic_power:<21} |
| Device Static (W)        | 2.946                 |
| Confidence Level         | High                  |
| Design Nets Matched      | {match_percent}% ({matched_nets}/{total_nets}) |
+--------------------------+-----------------------+
""".format(
        dynamic_power=dynamic_power,
        match_percent=match_percent,
        matched_nets=matched_nets,
        total_nets=total_nets,
    )


class ProjectValidationTests(unittest.TestCase):
    def test_accepts_classic_generated_project_and_discovers_top(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = create_project(Path(temporary), "custom_top")
            info = validate_project(project)
            self.assertEqual("custom_top", info.top)
            self.assertEqual("generated_project", info.identifier)
            self.assertEqual(64, len(info.project_hash))

    def test_rejects_nir_and_missing_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nir = root / "model.nir"
            nir.write_text("x", encoding="utf-8")
            with self.assertRaises(ValidationError):
                validate_project(nir)
            with self.assertRaises(ValidationError):
                validate_project(root)

    def test_hash_ignores_existing_vitis_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = create_project(Path(temporary))
            before = validate_project(project).project_hash
            result = project / "vitis_proj" / "sol" / "impl" / "output.dcp"
            result.parent.mkdir(parents=True)
            result.write_text("generated", encoding="utf-8")
            after = validate_project(project).project_hash
            self.assertEqual(before, after)

    def test_rejects_incomplete_medoid_testbench_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = create_project(Path(temporary))
            medoids = project / "tb_medoids"
            medoids.mkdir()
            (medoids / "testbench.cpp").write_text(
                "int main() { return 0; }\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValidationError, "Bundle de medoids incompleto"):
                validate_project(project)


class ConfigurationTests(unittest.TestCase):
    def test_default_saif_match_threshold_is_50_percent(self):
        self.assertEqual(50.0, load_config().saif_min_match_percent)

    def test_overrides_resolve_clock_once(self):
        config = load_config(
            overrides={
                "target": {"part": "xc7z020clg400-1", "clock": {"frequency_mhz": 100}},
                "power": {"saif_min_match_percent": 90},
            }
        )
        self.assertEqual("xc7z020clg400-1", config.part)
        self.assertEqual(10.0, config.clock_period_ns)
        self.assertEqual(90.0, config.saif_min_match_percent)

    def test_rejects_invalid_frequency(self):
        with self.assertRaises(ValidationError):
            load_config(overrides={"target": {"clock": {"frequency_mhz": 0}}})

    def test_rejects_negative_clock_uncertainty(self):
        with self.assertRaises(ValidationError):
            load_config(
                overrides={
                    "target": {
                        "clock": {
                            "frequency_mhz": 100,
                            "uncertainty_ns": -0.1,
                        }
                    }
                }
            )

    def test_rejects_invalid_clock_name_and_non_finite_values(self):
        for clock_name in ("", " ", "ap_*", "ap clk"):
            with self.subTest(clock_name=clock_name), self.assertRaises(ValidationError):
                load_config(overrides={"target": {"clock": {"name": clock_name}}})
        for field, value in (
            ("frequency_mhz", float("nan")),
            ("frequency_mhz", float("inf")),
            ("uncertainty_ns", float("nan")),
            ("uncertainty_ns", float("inf")),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                load_config(overrides={"target": {"clock": {field: value}}})

    def test_rejects_post_route_profile_until_it_is_implemented(self):
        with self.assertRaises(ValidationError):
            load_config(overrides={"simulation": {"profile": "power-accurate"}})

    def test_rejects_unsupported_saif_capture_scope(self):
        with self.assertRaises(ValidationError):
            load_config(
                overrides={
                    "simulation": {
                        "activity": {
                            "capture_scope": "testbench",
                        }
                    }
                }
            )


class SaifTests(unittest.TestCase):
    def test_parses_valid_saif(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity.saif"
            path.write_text(
                "(SAIFILE\n(TIMESCALE 1 ns)\n(DURATION 200)\n(INSTANCE dut (NET clk (T0 100) (T1 100) (TC 20)))\n)\n",
                encoding="utf-8",
            )
            info = parse_saif(path)
            self.assertEqual(200.0, info.duration)
            self.assertEqual(20, info.transition_count)

    def test_rejects_zero_duration_and_no_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity.saif"
            path.write_text("(SAIFILE (DURATION 0) (TC 0))", encoding="utf-8")
            with self.assertRaises(StageError):
                parse_saif(path)

    def test_rejects_header_only_saif_with_positive_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity.saif"
            path.write_text(
                "(SAIFILE\n(TIMESCALE 1 ps)\n(DURATION 1000)\n)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StageError, "transições"):
                parse_saif(path)

    def test_rejects_positive_duration_saif_with_only_zero_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity.saif"
            path.write_text(
                "(SAIFILE\n(DURATION 1000)\n(NET (clk (TC 0)))\n)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StageError, "transições"):
                parse_saif(path)


class ReportMetricsTests(unittest.TestCase):
    def test_parses_last_valid_csim_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "csim.log"
            log.write_text(
                " *** Final Acc: 87.50%\n"
                "diagnóstico intermediário\n"
                " *** Final Acc: 92.86%\n",
                encoding="utf-8",
            )
            self.assertEqual(92.86, parse_csim_accuracy(log))

            log.write_text("CSim done with 0 errors.\n", encoding="utf-8")
            self.assertIsNone(parse_csim_accuracy(log))

            log.write_text(" *** Final Acc: 101.00%\n", encoding="utf-8")
            self.assertIsNone(parse_csim_accuracy(log))

    def test_converts_saif_timescales_without_assuming_picoseconds(self):
        cases = (
            (1000, "1 ps", 1e-9),
            (1000, "10 ps", 1e-8),
            (200, "1 ns", 2e-7),
            (3, "1 us", 3e-6),
        )
        for duration, timescale, expected in cases:
            with self.subTest(timescale=timescale):
                self.assertAlmostEqual(
                    expected,
                    saif_duration_seconds(duration, timescale),
                )
        self.assertIsNone(saif_duration_seconds(100, None))
        self.assertIsNone(saif_duration_seconds(100, "ticks"))

    def test_parses_workload_and_uses_only_complete_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            testbench = Path(temporary) / "testbench.cpp"
            testbench.write_text(
                "\n".join(
                    (
                        "#define TOTAL_SAMPLES 15",
                        "#define BATCH_SIZE 14",
                        "#define STEP_COUNT 256",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            workload = parse_testbench_workload(testbench)
            self.assertIsNotNone(workload)
            assert workload is not None
            self.assertEqual(15, workload.declared_samples)
            self.assertEqual(14, workload.executed_samples)
            self.assertEqual(1, workload.batch_count)
            self.assertEqual(1, workload.ignored_samples)
            self.assertEqual(3584, workload.total_logical_steps)

    def test_returns_no_workload_when_required_macros_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            testbench = Path(temporary) / "testbench.cpp"
            testbench.write_text("int main() { return 0; }\n", encoding="utf-8")
            self.assertIsNone(parse_testbench_workload(testbench))

    def test_parses_vivado_power_summary_and_less_than_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "power.rpt"
            report.write_text(
                power_report_text(dynamic_power="<0.001"),
                encoding="utf-8",
            )
            info = parse_power_report(report)
            self.assertEqual(3.049, info.total_on_chip_power_w)
            self.assertEqual(0.001, info.dynamic_power_w)
            self.assertEqual("<0.001", info.dynamic_power_display)
            self.assertEqual(2.946, info.device_static_power_w)
            self.assertEqual("High", info.confidence_level)
            self.assertEqual(52.0, info.saif_match_percent)
            self.assertEqual(13439, info.saif_matched_design_nets)
            self.assertEqual(25972, info.saif_total_design_nets)

    def test_rejects_power_report_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "power.rpt"
            report.write_text(
                "| Design Nets Matched | 52% (1/2) |\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StageError, "campos obrigatórios"):
                parse_power_report(report)

    def test_derives_reference_latency_and_energy_metrics(self):
        workload = {
            "executed_samples": 14,
            "total_logical_steps": 3584,
        }
        activity = {
            "duration": 8525116350,
            "timescale": "1 ps",
        }
        power = {
            "total_on_chip_power_w": 3.049,
            "dynamic_power_w": 0.103,
            "device_static_power_w": 2.946,
            "saif_coverage_passed": False,
        }
        metrics = derive_summary_metrics(workload, activity, power)
        self.assertAlmostEqual(
            0.00852511635,
            metrics["capture_duration_seconds"],
        )
        self.assertAlmostEqual(
            2.3786596958705358e-6,
            metrics["average_latency_per_step_seconds"],
        )
        self.assertAlmostEqual(
            0.0006089368821428572,
            metrics["average_latency_per_sample_seconds"],
        )
        self.assertAlmostEqual(
            7.252533412709264e-6,
            metrics["energy_per_step_total_joules"],
        )
        self.assertAlmostEqual(
            0.0018566485536535716,
            metrics["energy_per_sample_total_joules"],
        )
        self.assertAlmostEqual(
            2.450019486746652e-7,
            metrics["energy_per_step_dynamic_joules"],
        )
        self.assertTrue(metrics["power_metrics_provisional"])


class VivadoUtilizationReportTests(unittest.TestCase):
    def test_parses_modern_device_report_and_validates_percentages(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "utilization_device_post_synth.rpt"
            report.write_text(utilization_report_text(), encoding="utf-8")
            info = parse_vivado_utilization_report(report, "xcu250-figd2104-2L-e")
            self.assertEqual("vivado_post_synth", info.source)
            self.assertEqual("Synthesized", info.design_state)
            self.assertEqual("out_of_context", info.mode)
            self.assertEqual("xcu250-figd2104-2L-e", info.part)
            self.assertEqual("xcu250-figd2104-2L-e", info.device)
            self.assertAlmostEqual(4945.0, info.resources["lut"].used)
            self.assertAlmostEqual(1728000.0, info.resources["lut"].available)
            self.assertAlmostEqual(0.2862, info.resources["lut"].utilization_percent)
            self.assertAlmostEqual(0.28616898148148145, info.resources["lut"].recalculated_utilization_percent)
            self.assertAlmostEqual(30.5, info.resources["bram"].used)
            self.assertAlmostEqual(2688.0, info.resources["bram"].available)
            self.assertAlmostEqual(1.1347, info.resources["bram"].utilization_percent)
            self.assertAlmostEqual(1.134672619047619, info.resources["bram"].recalculated_utilization_percent)
            self.assertIn("uram", info.resources)

    def test_parses_legacy_family_labels_without_uram(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "utilization_device_post_synth.rpt"
            report.write_text(
                utilization_report_text(
                    lut_label="Slice LUTs",
                    ff_label="Slice Registers",
                    include_uram=False,
                    include_prohibited=False,
                ),
                encoding="utf-8",
            )
            info = parse_vivado_utilization_report(report, "xcu250-figd2104-2L-e")
            self.assertNotIn("uram", info.resources)
            self.assertIn("lut", info.resources)
            self.assertIn("ff", info.resources)

    def test_rejects_part_divergence_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "utilization_device_post_synth.rpt"
            report.write_text(utilization_report_text(part="xczu3eg-sbva484-1-e"), encoding="utf-8")
            with self.assertRaisesRegex(StageError, "dispositivo diferente"):
                parse_vivado_utilization_report(report, "xcu250-figd2104-2L-e")

            report.write_text(
                utilization_report_text().replace("Block RAM Tile", "Missing RAM Tile"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StageError, "campos obrigatórios"):
                parse_vivado_utilization_report(report, "xcu250-figd2104-2L-e")


class TclRenderingTests(unittest.TestCase):
    def test_tcl_uses_custom_target_and_top(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(
                overrides={
                    "target": {
                        "part": "xc7z020clg400-1",
                        "clock": {"name": "clock_foo", "frequency_mhz": 100},
                    }
                }
            )
            rendered = vivado_synth_tcl(
                config,
                "custom_top",
                root / "rtl",
                [],
                root / "s.dcp",
                root / "netlist.v",
                root / "util.rpt",
                root / "util_device.rpt",
                root / "timing.rpt",
                root / "methodology.rpt",
            )
            self.assertIn("xc7z020clg400-1", rendered)
            self.assertIn("custom_top", rendered)
            self.assertIn("clock_foo", rendered)
            self.assertNotIn("xcu250-figd2104-2L-e", rendered)
            self.assertIn("report_utilization -hierarchical -file", rendered)
            self.assertIn("report_utilization -file", rendered)
            power = power_tcl(
                config,
                root / "s.dcp",
                root / "activity.saif",
                "tb/AESL_inst_custom_top",
                root / "power.rpt",
                root / "unmatched.rpt",
            )
            self.assertIn("tb/AESL_inst_custom_top", power)
            activity = xsim_saif_tcl(root / "activity.saif", "tb", "dut")
            self.assertIn("/tb/dut", activity)
            self.assertNotIn("log_saif [get_scopes", activity)
            scope_validation = "set capture_scopes [get_scopes -quiet $capture_scope]"
            exactly_one_scope = "if {[llength $capture_scopes] != 1}"
            save_scope = "set previous_scope [current_scope]"
            enter_scope = "current_scope [lindex $capture_scopes 0]"
            collect_objects = "set saif_objects [get_objects -r *]"
            open_saif = "open_saif "
            log_objects = "log_saif $saif_objects"
            restore_scope = "current_scope $previous_scope"
            run_all = "run all"
            close_saif = "close_saif"
            for command in (
                scope_validation,
                exactly_one_scope,
                save_scope,
                enter_scope,
                collect_objects,
                open_saif,
                log_objects,
                restore_scope,
                close_saif,
            ):
                self.assertIn(command, activity)
            self.assertLess(activity.index(scope_validation), activity.index(exactly_one_scope))
            self.assertLess(activity.index(exactly_one_scope), activity.index(save_scope))
            self.assertLess(activity.index(save_scope), activity.index(enter_scope))
            self.assertLess(activity.index(enter_scope), activity.index(collect_objects))
            self.assertLess(activity.index(collect_objects), activity.index(open_saif))
            self.assertLess(activity.index(open_saif), activity.index(log_objects))
            self.assertLess(activity.index(collect_objects), activity.index(log_objects))
            self.assertLess(activity.index(log_objects), activity.rindex(restore_scope))
            self.assertLess(activity.rindex(restore_scope), activity.index(run_all))
            self.assertLess(activity.index(log_objects), activity.index(run_all))
            self.assertLess(activity.index(run_all), activity.index(close_saif))
            self.assertIn("if {![llength $saif_objects]}", activity)

    def test_vivado_synth_uses_supported_include_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendered = vivado_synth_tcl(
                load_config(),
                "custom_top",
                root / "rtl",
                [],
                root / "s.dcp",
                root / "netlist.v",
                root / "util.rpt",
                root / "util_device.rpt",
                root / "timing.rpt",
                root / "methodology.rpt",
            )
            include_property = "set_property INCLUDE_DIRS $include_dirs [current_fileset]"
            read_verilog = "read_verilog -sv $hdl_files"
            self.assertIn(include_property, rendered)
            self.assertIn(read_verilog, rendered)
            self.assertLess(rendered.index(include_property), rendered.index(read_verilog))
            self.assertNotIn("-include_dirs", rendered)

    def test_vivado_synth_reads_clock_override_last_and_validates_after_synthesis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exported_xdc = root / "exported.xdc"
            clock_xdc = root / "clock_override.xdc"
            rendered = vivado_synth_tcl(
                load_config(),
                "custom_top",
                root / "rtl",
                [exported_xdc, clock_xdc],
                root / "s.dcp",
                root / "netlist.v",
                root / "util.rpt",
                root / "util_device.rpt",
                root / "timing.rpt",
                root / "methodology.rpt",
            )
            exported_read = "read_xdc {{{}}}".format(exported_xdc.resolve())
            clock_read = "read_xdc {{{}}}".format(clock_xdc.resolve())
            synthesis = "synth_design -top $top"
            validation = "set clock_port [get_ports -quiet $clock_name]"
            self.assertLess(rendered.index(exported_read), rendered.index(clock_read))
            self.assertLess(rendered.index(clock_read), rendered.index(synthesis))
            self.assertLess(rendered.index(synthesis), rendered.index(validation))
            self.assertNotIn("remove_clocks", rendered)
            self.assertNotIn("create_clock", rendered)

    def test_clock_override_xdc_uses_configured_clock(self):
        config = load_config(
            overrides={
                "target": {
                    "clock": {
                        "name": "clock_foo",
                        "frequency_mhz": 125,
                        "uncertainty_ns": 0.35,
                    }
                }
            }
        )
        rendered = clock_override_xdc(config)
        self.assertIn(
            "create_clock -name {clock_foo} -period 8 [get_ports {clock_foo}]",
            rendered,
        )
        self.assertIn("set_clock_uncertainty 0.35 [get_clocks {clock_foo}]", rendered)
        self.assertNotIn("\nif ", rendered)
        self.assertNotIn("remove_clocks", rendered)

    def test_power_uses_clock_preserved_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(
                overrides={
                    "target": {
                        "clock": {
                            "name": "clock_foo",
                            "frequency_mhz": 125,
                            "uncertainty_ns": 0.35,
                        }
                    }
                }
            )
            rendered = power_tcl(
                config,
                root / "s.dcp",
                root / "activity.saif",
                "tb/AESL_inst_custom_top",
                root / "power.rpt",
                root / "unmatched.rpt",
            )
            self.assertIn("set clock_name {clock_foo}", rendered)
            self.assertIn("set clock_period 8", rendered)
            self.assertIn("get_clocks -quiet -of_objects $clock_port", rendered)
            self.assertNotIn("remove_clocks", rendered)
            self.assertNotIn("create_clock", rendered)


class RunLayoutTests(unittest.TestCase):
    def test_cosim_setup_activates_medoid_bundle_after_full_csim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            full_testbench = project / "testbench.cpp"
            full_testbench.write_text(
                "#define TOTAL_SAMPLES 100\n#define BATCH_SIZE 10\n"
                "#define STEP_COUNT 32\nint main() { return 0; }\n",
                encoding="utf-8",
            )
            medoids = project / "tb_medoids"
            medoids.mkdir()
            medoid_testbench = (
                "#define TOTAL_SAMPLES 10\n#define BATCH_SIZE 10\n"
                "#define STEP_COUNT 32\nint main() { return 0; }\n"
            )
            (medoids / "testbench.cpp").write_text(
                medoid_testbench, encoding="utf-8"
            )
            (medoids / "data.txt").write_text("medoid-data\n", encoding="utf-8")
            (medoids / "targets.txt").write_text("medoid-targets\n", encoding="utf-8")

            pipeline = Pipeline.create(project, load_config(), root / "runs")
            pipeline._stage_prepare()
            self.assertIn("TOTAL_SAMPLES 100", (pipeline.project_dir / "testbench.cpp").read_text())

            with patch.object(pipeline, "_run_tool"):
                pipeline._stage_cosim_setup()

            self.assertEqual(
                medoid_testbench,
                (pipeline.project_dir / "testbench.cpp").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                "medoid-data\n",
                (pipeline.project_dir / "tb_data" / "data.txt").read_text(encoding="utf-8"),
            )
            artifact = pipeline.artifacts()["rtl_simulation_testbench"]
            self.assertEqual("medoids", artifact["profile"])
            self.assertEqual(10, artifact["workload"]["executed_samples"])

    def test_dry_run_records_requested_execution_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(
                project,
                load_config(),
                root / "runs",
                dry_run=True,
            )

            pipeline.run(to_stage="vivado-synth")

            expected_scope = {
                "from_stage": "prepare",
                "to_stage": "vivado-synth",
            }
            self.assertEqual(expected_scope, pipeline.status()["execution_scope"])
            summary = json.loads(
                (pipeline.reports_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(expected_scope, summary["execution_scope"])
            rendered = (pipeline.reports_dir / "summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("## Escopo da execução", rendered)
            self.assertIn("Etapa final: `vivado-synth`", rendered)
            self.assertNotIn("`post-synth-sim`: planned", rendered)
            self.assertIn("`post-synth-sim`: fora do escopo", rendered)
            self.assertIn("`power`: fora do escopo", rendered)

    def test_summary_records_csim_accuracy_and_missing_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            csim_log = pipeline.logs_dir / "csim.log"
            csim_log.write_text(
                " *** Final Acc: 88.57%\n",
                encoding="utf-8",
            )

            pipeline._write_summary()
            summary = json.loads(
                (pipeline.reports_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["csim"]["available"])
            self.assertEqual(88.57, summary["csim"]["accuracy_percent"])
            rendered = (pipeline.reports_dir / "summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("## Acurácia CSim", rendered)
            self.assertIn("Acurácia final: `88,57%`", rendered)
            self.assertIn("Log: `{}`".format(csim_log), rendered)

            csim_log.write_text("CSim sem resultado final\n", encoding="utf-8")
            pipeline._write_summary()
            summary = json.loads(
                (pipeline.reports_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(summary["csim"]["available"])
            self.assertIsNone(summary["csim"]["accuracy_percent"])
            rendered = (pipeline.reports_dir / "summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Acurácia final: `N/D`", rendered)

    def test_dry_run_creates_immutable_layout_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            source_contents = (project / "snn_implementation.cpp").read_text(encoding="utf-8")
            pipeline = Pipeline.create(project, load_config(), root / "runs", dry_run=True)
            pipeline.run()
            self.assertTrue((pipeline.run_dir / "run.yaml").is_file())
            self.assertEqual("dry-run", pipeline.status()["state"])
            self.assertEqual(source_contents, (project / "snn_implementation.cpp").read_text(encoding="utf-8"))

    def test_vivado_stage_creates_and_reads_clock_override_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            rtl_dir = pipeline.hls_solution_dir / "impl" / "ip" / "hdl" / "verilog"
            constraints_dir = pipeline.hls_solution_dir / "impl" / "ip" / "constraints"
            rtl_dir.mkdir(parents=True)
            constraints_dir.mkdir(parents=True)
            (rtl_dir / "custom_top.v").write_text("module custom_top; endmodule\n", encoding="utf-8")
            exported_xdc = constraints_dir / "exported.xdc"
            exported_xdc.write_text("# exported\n", encoding="utf-8")
            device_report = pipeline.run_dir / "40_vivado_synth" / "utilization_device_post_synth.rpt"
            device_report.write_text(utilization_report_text(), encoding="utf-8")

            with patch.object(pipeline, "_run_tool") as run_tool:
                pipeline._stage_vivado_synth()

            run_tool.assert_called_once()
            clock_xdc = pipeline.run_dir / "40_vivado_synth" / "clock_override.xdc"
            synth_tcl = pipeline.run_dir / "40_vivado_synth" / "synth_ooc.tcl"
            self.assertTrue(clock_xdc.is_file())
            rendered = synth_tcl.read_text(encoding="utf-8")
            exported_read = "read_xdc {{{}}}".format(exported_xdc.resolve())
            clock_read = "read_xdc {{{}}}".format(clock_xdc.resolve())
            self.assertLess(rendered.index(exported_read), rendered.index(clock_read))
            self.assertLess(rendered.index(clock_read), rendered.index("synth_design"))
            self.assertEqual(
                str(clock_xdc),
                pipeline.artifacts()["vivado_synth"]["clock_constraints"],
            )

    def test_vivado_synth_cache_requires_global_utilization_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            pipeline._mark_stage("vivado-synth", "success")
            synth_dir = pipeline.run_dir / "40_vivado_synth"
            synth_dir.mkdir(parents=True, exist_ok=True)
            (synth_dir / "post_synth.dcp").write_text("dcp\n", encoding="utf-8")
            (synth_dir / "post_synth_netlist.v").write_text("module top; endmodule\n", encoding="utf-8")
            (synth_dir / "utilization_post_synth.rpt").write_text("hier\n", encoding="utf-8")
            (synth_dir / "timing_post_synth.rpt").write_text("timing\n", encoding="utf-8")
            (synth_dir / "methodology_post_synth.rpt").write_text("method\n", encoding="utf-8")
            self.assertFalse(pipeline._stage_is_reusable("vivado-synth"))
            (synth_dir / "utilization_device_post_synth.rpt").write_text(
                utilization_report_text(),
                encoding="utf-8",
            )
            utilization_summary = parse_vivado_utilization_report(
                synth_dir / "utilization_device_post_synth.rpt",
                "xcu250-figd2104-2L-e",
            ).to_dict()
            (pipeline.reports_dir / "utilization_summary.json").write_text(
                json.dumps(utilization_summary),
                encoding="utf-8",
            )
            self.assertTrue(pipeline._stage_is_reusable("vivado-synth"))

    def test_low_coverage_preserves_power_and_generates_complete_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            (project / "testbench.cpp").write_text(
                "\n".join(
                    (
                        "#define TOTAL_SAMPLES 14",
                        "#define BATCH_SIZE 14",
                        "#define STEP_COUNT 256",
                        "int main() { return 0; }",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            # A política de cobertura pertence ao modo SAIF; o padrão do
            # ambiente é vectorless, que não anota atividade nenhuma.
            config = load_config(overrides={"power": {"activity_source": "saif"}})
            pipeline = Pipeline.create(project, config, root / "runs")
            pipeline._stage_prepare()
            saif = pipeline.run_dir / "60_activity" / "post_synth_activity.saif"
            saif.write_text(
                "(SAIFILE\n(TIMESCALE 1 ps)\n(DURATION 8525116350)"
                "\n(NET clk (TC 20))\n)\n",
                encoding="utf-8",
            )
            pipeline._set_artifact(
                "activity",
                {
                    "path": str(saif),
                    "duration": 8525116350,
                    "timescale": "1 ps",
                    "transition_count": 20,
                    "strip_path": "tb/dut",
                },
            )
            report = pipeline.run_dir / "70_power" / "power_report.rpt"
            report.write_text(
                power_report_text(match_percent="49"),
                encoding="utf-8",
            )

            with patch.object(pipeline, "_run_tool"):
                with self.assertRaisesRegex(StageError, "49.00% < 50.00%"):
                    pipeline._run_stage("power")

            power = pipeline.artifacts()["power"]
            self.assertFalse(power["saif_coverage_passed"])
            self.assertFalse(power["quality_accepted"])
            self.assertTrue(power["provisional"])
            self.assertEqual(3.049, power["total_on_chip_power_w"])
            self.assertEqual(0.103, power["dynamic_power_w"])
            self.assertEqual(2.946, power["device_static_power_w"])
            self.assertTrue(
                (pipeline.reports_dir / "power_summary.json").is_file()
            )

            pipeline._write_summary()
            summary = json.loads(
                (pipeline.reports_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("failed", summary["stages"]["power"]["state"])
            self.assertEqual(14, summary["workload"]["executed_samples"])
            self.assertEqual(3584, summary["workload"]["total_logical_steps"])
            self.assertAlmostEqual(
                7.252533412709264e-6,
                summary["derived_metrics"]["energy_per_step_total_joules"],
            )
            rendered = (pipeline.reports_dir / "summary.md").read_text(
                encoding="utf-8"
            )
            for expected in (
                "## Carga da simulação",
                "Amostras: `14`",
                "Passos temporais executados: `3.584`",
                "Duração simulada total (tempo lógico)",
                "`2,378660 µs/step`",
                "`608,936882 µs/amostra`",
                "## Potência e energia",
                "Potência total | `3,049 W`",
                "Potência dinâmica | `0,103 W`",
                "Potência estática | `2,946 W`",
                "`7,252533 µJ/step`",
                "`1,856649 mJ/amostra`",
                "estimativas provisórias",
            ):
                self.assertIn(expected, rendered)

    def test_summary_renders_hls_estimate_and_vivado_ooc_utilization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            device_report = pipeline.run_dir / "40_vivado_synth" / "utilization_device_post_synth.rpt"
            device_report.write_text(utilization_report_text(), encoding="utf-8")
            utilization_summary = parse_vivado_utilization_report(
                device_report,
                "xcu250-figd2104-2L-e",
            ).to_dict()
            pipeline._set_artifact(
                "vivado_synth",
                {
                    "checkpoint": str(pipeline.run_dir / "40_vivado_synth" / "post_synth.dcp"),
                    "netlist": str(pipeline.run_dir / "40_vivado_synth" / "post_synth_netlist.v"),
                    "utilization": str(pipeline.run_dir / "40_vivado_synth" / "utilization_post_synth.rpt"),
                    "utilization_hierarchical": str(pipeline.run_dir / "40_vivado_synth" / "utilization_post_synth.rpt"),
                    "utilization_device": str(device_report),
                    "timing": str(pipeline.run_dir / "40_vivado_synth" / "timing_post_synth.rpt"),
                    "methodology": str(pipeline.run_dir / "40_vivado_synth" / "methodology_post_synth.rpt"),
                    "rtl_dir": str(project / "vitis_proj" / "sol" / "impl" / "verilog"),
                    "clock_constraints": str(pipeline.run_dir / "40_vivado_synth" / "clock_override.xdc"),
                    "utilization_summary": utilization_summary,
                },
            )
            fake_hls_summary = {
                "report": str(root / "vitis_proj" / "sol" / "syn" / "report" / "snn_to_hls_csynth.xml"),
                "available": True,
                "resources": {
                    "LUT": "123",
                    "FF": "456",
                    "BRAM_18K": "7",
                    "DSP": "8",
                    "URAM": "0",
                },
                "latency": {
                    "best_case": "10",
                    "average_case": "11",
                    "worst_case": "12",
                },
            }
            with patch.object(pipeline, "_hls_summary", return_value=fake_hls_summary):
                pipeline._write_summary()
            rendered = (pipeline.reports_dir / "summary.md").read_text(encoding="utf-8")
            for expected in (
                "## Estimativa de recursos HLS",
                "Valores reportados pelo csynth.xml",
                "## Uso de recursos pós-síntese — Vivado OOC",
                "| LUT | `4.945` | `1.728.000` | `0,286%` |",
                "| BRAM | `30,5` | `2.688` | `1,135%` |",
                "Os valores estruturados preservam o percentual informado pelo Vivado",
                "Latência HLS: melhor caso `10`, média `11` e pior caso `12`.",
            ):
                self.assertIn(expected, rendered)

    def test_disabled_coverage_gate_keeps_low_match_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            config = load_config(
                overrides={
                    "power": {
                        "saif_min_match_percent": 95,
                        "fail_on_low_confidence": False,
                    }
                }
            )
            pipeline = Pipeline.create(project, config, root / "runs")
            report = pipeline.run_dir / "70_power" / "power_report.rpt"
            unmatched = pipeline.run_dir / "70_power" / "unmatched.rpt"
            report.write_text(power_report_text(), encoding="utf-8")

            power = pipeline._record_power_report(report, unmatched)

            self.assertFalse(power["saif_coverage_passed"])
            self.assertFalse(power["coverage_policy_enforced"])
            self.assertTrue(power["quality_accepted"])
            self.assertTrue(power["provisional"])

    def test_post_synth_project_rebases_retained_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root, "custom_top")
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            source_dir = root / "hls_sim"
            output_dir = root / "post_synth"
            (source_dir / "svtb").mkdir(parents=True)
            (source_dir / "custom_top_subsystem").mkdir()
            output_dir.mkdir()
            retained = (
                source_dir / "glbl.v",
                source_dir / "AESL_automem_input_r.v",
                source_dir / "svtb" / "sv_module_top.sv",
                source_dir
                / "custom_top_subsystem"
                / "custom_top_subsystem_pkg.sv",
            )
            for path in retained:
                path.write_text("module placeholder; endmodule\n", encoding="utf-8")
            source_prj = source_dir / "custom_top.prj"
            source_prj.write_text(
                "\n".join(
                    (
                        'sv xil_defaultlib "glbl.v"',
                        'sv xil_defaultlib "AESL_automem_input_r.v"',
                        'sv xil_defaultlib "custom_top.autotb.v"',
                        'sv xil_defaultlib "custom_top.v"',
                        'sv xil_defaultlib "dataflow_monitor.sv"',
                        'sv xil_defaultlib "./svtb/sv_module_top.sv"',
                        'sv xil_defaultlib "custom_top_subsystem/custom_top_subsystem_pkg.sv"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output_prj = output_dir / "post_synth.prj"
            post_autotb = output_dir / "post_synth.autotb.v"
            netlist = output_dir / "post_synth_netlist.v"
            stubs = output_dir / "monitor_stubs.sv"

            pipeline._post_synth_prj(
                source_prj,
                output_prj,
                post_autotb,
                netlist,
                stubs,
            )

            rendered = output_prj.read_text(encoding="utf-8")
            for path in retained:
                self.assertIn('"{}"'.format(path.resolve()), rendered)
            self.assertIn('"{}"'.format(post_autotb.resolve()), rendered)
            self.assertIn('"{}"'.format(netlist.resolve()), rendered)
            self.assertIn('"{}"'.format(stubs.resolve()), rendered)
            self.assertNotIn('"custom_top.v"', rendered)
            self.assertNotIn("dataflow_monitor.sv", rendered)

    def test_post_synth_project_rejects_missing_retained_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root, "custom_top")
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            source_dir = root / "hls_sim"
            output_dir = root / "post_synth"
            source_dir.mkdir()
            output_dir.mkdir()
            source_prj = source_dir / "custom_top.prj"
            source_prj.write_text(
                'sv xil_defaultlib "glbl.v"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(StageError, "glbl.v"):
                pipeline._post_synth_prj(
                    source_prj,
                    output_dir / "post_synth.prj",
                    output_dir / "post_synth.autotb.v",
                    output_dir / "post_synth_netlist.v",
                    output_dir / "monitor_stubs.sv",
                )

    def test_post_synth_project_does_not_compile_glbl_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root, "custom_top")
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            source_dir = root / "hls_sim"
            output_dir = root / "post_synth"
            source_dir.mkdir()
            output_dir.mkdir()
            source_glbl = source_dir / "glbl.v"
            source_glbl.write_text("module glbl; endmodule\n", encoding="utf-8")
            source_prj = source_dir / "custom_top.prj"
            source_prj.write_text('sv xil_defaultlib "glbl.v"\n', encoding="utf-8")
            netlist = output_dir / "post_synth_netlist.v"
            netlist.write_text("module custom_top; endmodule\nmodule glbl; endmodule\n", encoding="utf-8")
            output_prj = output_dir / "post_synth.prj"

            pipeline._post_synth_prj(
                source_prj,
                output_prj,
                output_dir / "post_synth.autotb.v",
                netlist,
                output_dir / "monitor_stubs.sv",
            )

            rendered = output_prj.read_text(encoding="utf-8")
            self.assertNotIn(str(source_glbl.resolve()), rendered)
            self.assertEqual(1, rendered.count(str(netlist.resolve())))

    def test_failure_refreshes_summary_without_masking_original_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            with patch.object(
                pipeline,
                "_stage_prepare",
                side_effect=StageError("falha deliberada"),
            ):
                with self.assertRaisesRegex(StageError, "falha deliberada"):
                    pipeline.run(to_stage="prepare")

            summary = json.loads(
                (pipeline.run_dir / "reports" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["state"])
            self.assertEqual("failed", summary["stages"]["prepare"]["state"])

    def test_retry_clears_stale_stage_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            pipeline._mark_stage("prepare", "failed", error="erro anterior")
            self.assertIn("finished_at", pipeline.status())
            self.assertIn("finished_at", pipeline.status()["stages"]["prepare"])
            pipeline._mark_stage("prepare", "running", started_at="agora")
            self.assertNotIn("finished_at", pipeline.status())
            pipeline._mark_stage("prepare", "success", finished_at="depois")

            entry = pipeline.status()["stages"]["prepare"]
            self.assertEqual("success", entry["state"])
            self.assertNotIn("error", entry)
            self.assertNotIn("reason", entry)

    def test_planned_and_skipped_states_clear_stale_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            pipeline._mark_stage(
                "prepare",
                "failed",
                error="erro anterior",
                started_at="antes",
            )
            pipeline._mark_stage("prepare", "planned", reason="dry-run")
            planned = pipeline.status()["stages"]["prepare"]
            self.assertNotIn("error", planned)
            self.assertNotIn("started_at", planned)
            self.assertNotIn("finished_at", planned)

            pipeline._mark_stage(
                "prepare",
                "failed",
                error="outro erro",
                started_at="antes",
            )
            pipeline._mark_stage("prepare", "skipped", reason="cache")
            skipped = pipeline.status()["stages"]["prepare"]
            self.assertNotIn("error", skipped)
            self.assertNotIn("started_at", skipped)
            self.assertNotIn("finished_at", skipped)

    def test_interruption_marks_stage_failed_and_refreshes_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            with patch.object(
                pipeline,
                "_stage_prepare",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.run(to_stage="prepare")

            entry = pipeline.status()["stages"]["prepare"]
            self.assertEqual("failed", entry["state"])
            self.assertEqual("Execução interrompida pelo usuário", entry["error"])
            summary = json.loads(
                (pipeline.run_dir / "reports" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["state"])
            self.assertEqual("failed", summary["stages"]["prepare"]["state"])


class CommandRunnerTests(unittest.TestCase):
    def test_keyboard_interrupt_terminates_process_group_and_is_reraised(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock()
            process.stdout = iter(())
            process.pid = 4242
            output_queue = Mock()
            output_queue.get.side_effect = KeyboardInterrupt

            with (
                patch("sim.utils.subprocess.Popen", return_value=process) as popen,
                patch("sim.utils.queue.Queue", return_value=output_queue),
                patch("sim.utils._terminate_process_group") as terminate,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    CommandRunner().run(
                        ["fake-tool"],
                        cwd=root,
                        log_path=root / "tool.log",
                    )

            terminate.assert_called_once_with(process)
            if os.name != "nt":
                self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_timeout_terminates_process_group_before_error_124(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock()
            process.stdout = iter(())
            process.pid = 4242
            output_queue = Mock()
            output_queue.get.side_effect = queue.Empty

            with (
                patch("sim.utils.subprocess.Popen", return_value=process),
                patch("sim.utils.queue.Queue", return_value=output_queue),
                patch("sim.utils._terminate_process_group") as terminate,
                patch("sim.utils.time.monotonic", side_effect=(0.0, 2.0)),
            ):
                with self.assertRaises(CommandError) as raised:
                    CommandRunner(timeout_seconds=1).run(
                        ["fake-tool"],
                        cwd=root,
                        log_path=root / "tool.log",
                    )

            self.assertEqual(124, raised.exception.returncode)
            terminate.assert_called_once_with(process)
            self.assertIn(
                "Processo excedeu o timeout",
                (root / "tool.log").read_text(encoding="utf-8"),
            )

    @unittest.skipUnless(os.name == "posix", "encerramento por grupo é POSIX")
    def test_posix_process_group_termination_escalates_after_grace_period(self):
        process = Mock()
        process.pid = 4242
        process.wait.side_effect = (
            subprocess.TimeoutExpired(["fake-tool"], 1),
            0,
        )
        with (
            patch("sim.utils.os.killpg") as killpg,
            patch("sim.utils._process_group_exists", return_value=True),
            patch("sim.utils.time.monotonic", side_effect=(0.0, 2.0)),
        ):
            _terminate_process_group(process, grace_seconds=1)

        self.assertEqual(
            [
                call(4242, signal.SIGTERM),
                call(4242, signal.SIGKILL),
            ],
            killpg.call_args_list,
        )
        self.assertEqual(2, process.wait.call_count)


class VectorlessPowerTests(unittest.TestCase):
    """Modo vectorless: sem SAIF, sem simulação pós-síntese, sem cobertura."""

    def test_default_activity_source_is_vectorless(self):
        self.assertEqual("vectorless", load_config().activity_source)

    def test_invalid_activity_source_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "activity_source"):
            load_config(overrides={"power": {"activity_source": "medido"}})

    def test_vectorless_tcl_declares_rates_and_omits_saif(self):
        script = power_tcl(
            load_config(), Path("/tmp/post_synth.dcp"), None, None,
            Path("/tmp/power.rpt"), None,
        )
        self.assertNotIn("read_saif", script)
        self.assertIn("set_switching_activity -default_toggle_rate 12.5", script)
        self.assertIn("-default_static_probability 0.5", script)
        self.assertIn("report_power -file", script)

    def test_saif_tcl_still_annotates(self):
        script = power_tcl(
            load_config(overrides={"power": {"activity_source": "saif"}}),
            Path("/tmp/post_synth.dcp"), Path("/tmp/a.saif"), "tb/dut",
            Path("/tmp/power.rpt"), Path("/tmp/unmatched.rpt"),
        )
        self.assertIn("read_saif -strip_path", script)
        self.assertNotIn("set_switching_activity", script)

    def test_power_report_without_saif_row_parses(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "power_report.rpt"
            text = power_report_text(match_percent="49")
            # Remove a linha que só existe quando um SAIF foi lido.
            text = "\n".join(
                line for line in text.splitlines()
                if "Design Nets Matched" not in line
            ) + "\n"
            report.write_text(text, encoding="utf-8")
            info = parse_power_report(report)
            self.assertIsNone(info.saif_match_percent)
            self.assertEqual(3.049, info.total_on_chip_power_w)

    def test_saif_stages_are_not_required_when_vectorless(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            for stage in ("cosim-setup", "post-synth-sim"):
                self.assertFalse(pipeline._stage_is_required(stage))
            for stage in ("prepare", "vivado-synth", "power"):
                self.assertTrue(pipeline._stage_is_required(stage))

            saif_pipeline = Pipeline.create(
                project,
                load_config(overrides={"power": {"activity_source": "saif"}}),
                root / "runs_saif",
            )
            for stage in ("cosim-setup", "post-synth-sim"):
                self.assertTrue(saif_pipeline._stage_is_required(stage))

    def test_vectorless_power_stage_skips_saif_and_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = create_project(root)
            pipeline = Pipeline.create(project, load_config(), root / "runs")
            pipeline._stage_prepare()
            report = pipeline.run_dir / "70_power" / "power_report.rpt"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(power_report_text(match_percent="49"), encoding="utf-8")

            with patch.object(pipeline, "_run_tool"):
                pipeline._run_stage("power")

            power = pipeline.artifacts()["power"]
            self.assertEqual("vectorless", power["activity_source"])
            self.assertIsNone(power["saif_coverage_passed"])
            self.assertTrue(power["quality_accepted"])
            # Taxas padrão não são a atividade do workload: nunca definitivo.
            self.assertTrue(power["provisional"])
            self.assertTrue(
                (pipeline.run_dir / "70_power" / "power_vectorless.tcl").is_file()
            )

    def test_energy_falls_back_to_hls_latency_window(self):
        workload = {"executed_samples": 10, "total_logical_steps": 320}
        power = {
            "total_on_chip_power_w": 2.976,
            "dynamic_power_w": 0.031,
            "device_static_power_w": 2.945,
        }
        metrics = derive_summary_metrics(
            workload, None, power, analytic_duration_seconds=0.2177344,
        )
        self.assertEqual(
            "hls_latency_times_logical_steps", metrics["latency_definition"]
        )
        self.assertAlmostEqual(0.2177344, metrics["capture_duration_seconds"])
        self.assertAlmostEqual(
            2.976 * 0.2177344, metrics["capture_energy_total_joules"]
        )

    def test_saif_window_wins_over_analytic_window(self):
        metrics = derive_summary_metrics(
            {"executed_samples": 10, "total_logical_steps": 320},
            {"duration_seconds": 0.21749766795, "timescale": "1 ps"},
            None,
            analytic_duration_seconds=0.2177344,
        )
        self.assertEqual(
            "full_saif_window_amortized", metrics["latency_definition"]
        )
        self.assertAlmostEqual(0.21749766795, metrics["capture_duration_seconds"])


if __name__ == "__main__":
    unittest.main()
