import subprocess
import os
import shutil
import re
import shlex
from pathlib import Path

from .backend_utils import copy_backend_to
from .read_nir import get_model_config_from_nir
from .testbench_manager import TestbenchManager
from .implementation_manager import implement_model

class NeuroHls:

    def __init__(self, folder_path: str, settings64_path: str = None):

        self._folder_path = folder_path
        self._settings64_path = settings64_path

        if not os.path.exists(folder_path):
            copy_backend_to(folder_path)

        self._has_parsed_nir = False
        self._has_created_testbench = False

        self._tb_manager = TestbenchManager(folder_path)

        self._project_name = "vitis_proj"

    def read_nir_file(self, nir_file_path, metadata_file_path = None):

        model = get_model_config_from_nir(nir_file_path, metadata_file_path)
        self._input_shape = model.input_shape
        self._output_shape = model.output_shape

        self._has_parsed_nir = True
        
        return model 

    def implement_model(self, model, use_float, use_event_driven = False):

        implement_model(model, self._folder_path, use_float, use_event_driven)

    def define_test_dataset(self, dataset_file_path: str, data_is_binary: bool, step_count: int, different_sample_per_step: bool):
        
        self._tb_manager.define_dataset(dataset_file_path, data_is_binary, step_count, different_sample_per_step)

    def create_testbench(self, total_samples: int, batch_size: int, reset_potentials = False, debug_mode = False):

        if not self._has_parsed_nir:
            print("ERROR: The network architecture must be defined before creating the testbench files.")
            return
        
        used_total_samples, used_batch_size = self._tb_manager.define_sample_count_and_batch_size(total_samples, batch_size)

        print(f"Total samples used: {used_total_samples} of {self._tb_manager.get_number_of_available_samples()}")
        print(f"Batch size: {used_batch_size}")
        print(f"Total batches: {used_total_samples // used_batch_size}")
        
        self._tb_manager.create_testbench_file(self._input_shape, self._output_shape[0], reset_potentials, debug_mode)
        self._has_created_testbench = True

        print("Testbench was created.")

    def _run_vitis_command(self, args):
        if self._settings64_path and os.name == 'nt':
            command_str = " ".join(args)
            cmd = f'call "{self._settings64_path}" && {command_str}'
            process = subprocess.Popen(cmd, shell=True, cwd=self._folder_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()
        else:
            process = subprocess.Popen(args, cwd=self._folder_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()

    def _write_shell_script(self, script_path: Path, commands, cwd: Path = None):
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("set -e\n")
            if cwd is not None:
                f.write(f"cd {shlex.quote(cwd.resolve().as_posix())}\n")
            for command in commands:
                f.write(" ".join(shlex.quote(str(arg)) for arg in command))
                f.write("\n")
        script_path.chmod(0o755)

    def _create_vitis_project_if_needed(self, force_reset: bool = False):

        proj_path = Path(self._folder_path) / self._project_name

        if os.path.exists(proj_path) and not force_reset:
            return

        if os.path.exists(proj_path):
            shutil.rmtree(proj_path)
        
        print("Creating Vitis Project...\n")
        
        if not self._tb_manager.is_ready():
            print("\n*** Project creation aborted.")
            raise Exception("Missing file")

        tcl_wrapper = f'set argv [list "{self._project_name}"]\nset argc 1\nsource 0_create_project.tcl\n'
        with open(Path(self._folder_path) / "wrapper_create_project.tcl", "w") as f: f.write(tcl_wrapper)
        self._run_vitis_command(["vitis-run", "--mode", "hls", "--tcl", "wrapper_create_project.tcl"])

    def _clean_solution_sim_artifacts(self, solution_name: str = "sol"):

        sim_path = Path(self._folder_path) / self._project_name / solution_name / "sim"

        if sim_path.exists():
            shutil.rmtree(sim_path)

    def run_csim(self, solution_name = "sol"):
        
        try:
            self._create_vitis_project_if_needed()
            tcl_wrapper = f'set argv [list "{self._project_name}" "{solution_name}"]\nset argc 2\nsource 1_csim.tcl\n'
            with open(Path(self._folder_path) / "wrapper_csim.tcl", "w") as f: f.write(tcl_wrapper)
            self._run_vitis_command(["vitis-run", "--mode", "hls", "--tcl", "wrapper_csim.tcl"])
        except Exception as e:
            print(f"\n*** Unable to run C-Simulation: {e}")
            raise

    def run_synth(self, frequency_MHz: int, part = "xc7z020clg400-1", solution_name = "sol"):

        if frequency_MHz <= 0:
            print("ERROR: frequency must be greater than 0.\n")
            return

        clk_period_ns = 1000 / frequency_MHz

        try:
            self._create_vitis_project_if_needed()
            tcl_wrapper = f'set argv [list "{self._project_name}" "{solution_name}" "{clk_period_ns}" "{part}"]\nset argc 4\nsource 2_synth.tcl\n'
            with open(Path(self._folder_path) / "wrapper_synth.tcl", "w") as f: f.write(tcl_wrapper)
            self._run_vitis_command(["vitis-run", "--mode", "hls", "--tcl", "wrapper_synth.tcl"])
        except Exception:
            print("\n*** Unable to run Synthesis.")
    
    def run_cosim(self, solution_name = "sol", setup_only=False):

        try:
            self._create_vitis_project_if_needed()
            if setup_only:
                tcl_wrapper = f'set argv [list "{self._project_name}" "{solution_name}"]\nset argc 2\nopen_project "{self._project_name}"\nopen_solution "{solution_name}"\ncosim_design -setup\nexit\n'
            else:
                tcl_wrapper = f'set argv [list "{self._project_name}" "{solution_name}"]\nset argc 2\nopen_project "{self._project_name}"\nopen_solution "{solution_name}"\ncosim_design\nexit\n'
            with open(Path(self._folder_path) / "wrapper_cosim.tcl", "w") as f: f.write(tcl_wrapper)
            self._run_vitis_command(["vitis-run", "--mode", "hls", "--tcl", "wrapper_cosim.tcl"])
        except Exception as e:
            print(f"\n*** Unable to run Co-Simulation: {e}")

    def run_cosim_saif_capture(self, solution_name = "sol"):
        self._build_cosim_saif_artifacts(solution_name)

    def run_export_design(self, solution_name = "sol"):
        self._export_component_to_vivado(solution_name)

    def _export_component_to_vivado(self, solution_name = "sol"):

        try:
            self._create_vitis_project_if_needed()
            tcl_wrapper = f'set argv [list "{self._project_name}" "{solution_name}"]\nset argc 2\nsource 4_export.tcl\n'
            with open(Path(self._folder_path) / "wrapper_export.tcl", "w") as f: f.write(tcl_wrapper)
            self._run_vitis_command(["vitis-run", "--mode", "hls", "--tcl", "wrapper_export.tcl"])
            vivado_script = Path(self._folder_path) / self._project_name / solution_name / "impl" / "verilog" / "run_power_report.tcl"
            vivado_ip_dir = (Path(self._folder_path) / self._project_name / solution_name / "impl" / "ip" / "hdl" / "verilog").resolve()
            vivado_xdc = (Path(self._folder_path) / self._project_name / solution_name / "impl" / "ip" / "constraints" / "snn_to_hls_ooc.xdc").resolve()
            vivado_wdb = (Path(self._folder_path) / self._project_name / solution_name / "sim" / "verilog" / "snn_to_hls.wdb").resolve()
            power_report = (Path(self._folder_path) / self._project_name / solution_name / "impl" / "verilog" / "power_report.txt").resolve()
            vivado_script.parent.mkdir(parents=True, exist_ok=True)
            with open(vivado_script, "w", encoding="utf-8") as f:
                f.write("create_project -in_memory -part xcu250-figd2104-2L-e\n")
                f.write(f"set vivado_ip_dir {vivado_ip_dir.as_posix()}\n")
                f.write("set hdl_files [glob -nocomplain -directory $vivado_ip_dir *.v]\n")
                f.write("if {![llength $hdl_files]} { error \"No exported HLS HDL files were found\" }\n")
                f.write("set_property include_dirs $vivado_ip_dir [current_fileset]\n")
                f.write("read_verilog -sv $hdl_files\n")
                if vivado_xdc.is_file():
                    f.write(f"read_xdc {vivado_xdc.as_posix()}\n")
                f.write("synth_design -top snn_to_hls -part xcu250-figd2104-2L-e\n")
                if vivado_wdb.is_file():
                    f.write(f"open_wave_database {vivado_wdb.as_posix()}\n")
                f.write(f"report_power -file {power_report.as_posix()} -name {{power_1}}\n")
                f.write("close_project\n")
            self._run_vitis_command(["vivado", "-mode", "batch", "-source", str(vivado_script.resolve())])
        except Exception as e:
            print(f"\n*** Unable to run Export Design: {e}")

    def run_post_synth_saif_capture(self, solution_name = "sol"):
        self._build_post_synth_saif_artifacts(solution_name)

    def _build_cosim_saif_artifacts(self, solution_name = "sol"):

        try:
            self._create_vitis_project_if_needed()
            self._clean_solution_sim_artifacts(solution_name)
            self.run_cosim(solution_name)

            base_dir = Path(self._folder_path) / self._project_name / solution_name
            impl_verilog_dir = base_dir / "impl" / "verilog"
            sim_verilog_dir = base_dir / "sim" / "verilog"
            cosim_wdb = (sim_verilog_dir / "snn_to_hls.wdb").resolve()
            cosim_saif = (impl_verilog_dir / "cosim_activity.saif").resolve()
            cosim_tcl = (sim_verilog_dir / "run_cosim_saif.tcl").resolve()
            cosim_bat = (sim_verilog_dir / "run_cosim_saif.bat").resolve()
            cosim_sh = (sim_verilog_dir / "run_cosim_saif.sh").resolve()

            if not cosim_wdb.is_file():
                raise FileNotFoundError(f"Missing cosimulation WDB file: {cosim_wdb}")

            impl_verilog_dir.mkdir(parents=True, exist_ok=True)
            sim_verilog_dir.mkdir(parents=True, exist_ok=True)

            with open(cosim_tcl, "w", encoding="utf-8") as f:
                f.write(f"open_saif {cosim_saif.as_posix()}\n")
                f.write("log_saif [get_scopes]\n")
                f.write("run all\n")
                f.write("close_saif\n")
                f.write("quit\n")

            xsim_args = [
                "xsim",
                "-testplusarg", "UVM_VERBOSITY=UVM_NONE",
                "-testplusarg", "UVM_TESTNAME=snn_to_hls_test_lib",
                "-testplusarg", "UVM_TIMEOUT=20000000000000",
                "--noieeewarnings",
                "snn_to_hls",
                "-tclbatch", cosim_tcl.as_posix(),
            ]

            if os.name == "nt":
                with open(cosim_bat, "w", encoding="utf-8") as f:
                    f.write("@echo off\r\n")
                    f.write(f'pushd "{sim_verilog_dir.as_posix()}"\r\n')
                    f.write("call C:/AMDDesignTools/2025.2/Vivado/bin/" + " ".join(f'"{arg}"' if " " in str(arg) else str(arg) for arg in xsim_args) + "\r\n")
                    f.write("popd\r\n")
                self._run_vitis_command([str(cosim_bat)])
            else:
                self._write_shell_script(cosim_sh, [xsim_args], cwd=sim_verilog_dir)
                self._run_vitis_command(["bash", str(cosim_sh)])
        except Exception as e:
            print(f"\n*** Unable to build cosimulation SAIF artifacts: {e}")

    def configure_cosim_and_export(self, solution_name="sol"):
        try:
            print("\n*** [Step 1] Configuring Co-simulation (RTL Setup only) and Exporting ***")
            self.run_cosim(solution_name, setup_only=True)
            self._create_vitis_project_if_needed()

            tcl_wrapper = f'set argv [list "{self._project_name}" "{solution_name}"]\nset argc 2\nsource 4_export.tcl\n'
            with open(Path(self._folder_path) / "wrapper_export.tcl", "w") as f:
                f.write(tcl_wrapper)

            self._run_vitis_command(["vitis-run", "--mode", "hls", "--tcl", "wrapper_export.tcl"])
        except Exception as e:
            print(f"\n*** Unable to configure cosim and export: {e}")

    def run_vivado_synthesis(self, solution_name="sol"):
        try:
            print("\n*** [Step 2] Running Vivado Synthesis ***")
            base_dir = Path(self._folder_path) / self._project_name / solution_name
            impl_verilog_dir = base_dir / "impl" / "verilog"
            sim_verilog_dir = base_dir / "sim" / "verilog"
            vivado_ip_dir = (base_dir / "impl" / "ip" / "hdl" / "verilog").resolve()
            vivado_xdc = (base_dir / "impl" / "ip" / "constraints" / "snn_to_hls_ooc.xdc").resolve()
            vivado_synth_script = (impl_verilog_dir / "run_ooc_synth.tcl").resolve()
            vivado_checkpoint = (impl_verilog_dir / "post_synth.dcp").resolve()
            vivado_netlist = (sim_verilog_dir / "post_synth_netlist.v").resolve()
            util_report = (impl_verilog_dir / "utilization_report.txt").resolve()

            impl_verilog_dir.mkdir(parents=True, exist_ok=True)
            sim_verilog_dir.mkdir(parents=True, exist_ok=True)

            with open(vivado_synth_script, "w", encoding="utf-8") as f:
                f.write("create_project -in_memory -part xcu250-figd2104-2L-e\n")
                f.write(f"set vivado_ip_dir {vivado_ip_dir.as_posix()}\n")
                f.write("set hdl_files [glob -nocomplain -directory $vivado_ip_dir *.v]\n")
                f.write("if {![llength $hdl_files]} { error \"No exported HLS HDL files were found\" }\n")
                f.write("set_property include_dirs $vivado_ip_dir [current_fileset]\n")
                f.write("read_verilog -sv $hdl_files\n")
                if vivado_xdc.is_file():
                    f.write(f"read_xdc {vivado_xdc.as_posix()}\n")
                f.write("synth_design -top snn_to_hls -part xcu250-figd2104-2L-e -mode out_of_context -flatten_hierarchy none -keep_equivalent_registers\n")
                f.write(f"report_utilization -file {util_report.as_posix()}\n")
                f.write(f"write_verilog -force -mode funcsim {vivado_netlist.as_posix()}\n")
                f.write(f"write_checkpoint -force {vivado_checkpoint.as_posix()}\n")
                f.write("close_project\n")

            self._run_vitis_command(["vivado", "-mode", "batch", "-source", str(vivado_synth_script)])
        except Exception as e:
            print(f"\n*** Unable to run vivado synthesis: {e}")

    def generate_post_synth_saif(self, solution_name="sol"):
        try:
            print("\n*** [Step 3] Generating Post-Synth SAIF Simulation ***")
            base_dir = Path(self._folder_path) / self._project_name / solution_name
            impl_verilog_dir = base_dir / "impl" / "verilog"
            sim_verilog_dir = base_dir / "sim" / "verilog"
            vivado_netlist = (sim_verilog_dir / "post_synth_netlist.v").resolve()
            saif_file = (impl_verilog_dir / "post_synth_activity.saif").resolve()
            post_synth_prj = (sim_verilog_dir / "post_synth.prj").resolve()
            post_synth_tcl = (sim_verilog_dir / "post_synth_saif.tcl").resolve()
            post_synth_autotb = (sim_verilog_dir / "post_synth.autotb.v").resolve()

            source_prj = sim_verilog_dir / "snn_to_hls.prj"
            if not source_prj.is_file():
                raise FileNotFoundError(f"Missing simulation project file: {source_prj}")

            source_autotb = sim_verilog_dir / "snn_to_hls.autotb.v"
            if not source_autotb.is_file():
                raise FileNotFoundError(f"Missing autogenerated testbench file: {source_autotb}")

            with open(source_autotb, "r", encoding="utf-8") as f:
                autotb_lines = f.readlines()

            stripped_autotb_lines = []
            skipping_dataflow_monitor = False
            for line in autotb_lines:
                if not skipping_dataflow_monitor and "dataflow status monitor" in line:
                    skipping_dataflow_monitor = True
                    continue

                if skipping_dataflow_monitor:
                    if line.strip() == "endmodule":
                        stripped_autotb_lines.append(line)
                        skipping_dataflow_monitor = False
                    continue

                stripped_autotb_lines.append(line)

            with open(post_synth_autotb, "w", encoding="utf-8") as f:
                f.writelines(stripped_autotb_lines)

            with open(source_prj, "r", encoding="utf-8") as f:
                prj_lines = f.readlines()

            custom_prj_lines = []
            inserted_netlist = False
            netlist_line = f'sv xil_defaultlib "{vivado_netlist.as_posix()}"\n'

            for line in prj_lines:
                stripped_line = line.strip()

                if not stripped_line or stripped_line.startswith("#"):
                    custom_prj_lines.append(line)
                    continue

                if "dataflow_monitor.sv" in stripped_line:
                    continue

                if "snn_to_hls.autotb.v" in stripped_line:
                    custom_prj_lines.append(f'sv xil_defaultlib "{post_synth_autotb.as_posix()}"\n')
                    continue

                if re.search(r'"snn_to_hls(?!\.autotb\.v)', stripped_line):
                    continue

                custom_prj_lines.append(line)

                if (not inserted_netlist) and ("glbl.v" in stripped_line):
                    custom_prj_lines.append(netlist_line)
                    inserted_netlist = True

            if not inserted_netlist:
                custom_prj_lines.append(netlist_line)

            with open(post_synth_prj, "w", encoding="utf-8") as f:
                f.writelines(custom_prj_lines)

            with open(post_synth_tcl, "w", encoding="utf-8") as f:
                f.write("load_features simulator\n")
                f.write(f"open_saif {saif_file.as_posix()}\n")
                f.write("log_saif [get_scopes -r /apatb_snn_to_hls_top]\n")
                f.write("run all\n")
                f.write("close_saif\n")
                f.write("quit\n")

            xelab_args = [
                "xelab",
                "xil_defaultlib.apatb_snn_to_hls_top",
                "xil_defaultlib.glbl",
                "-Oenable_linking_all_libraries",
                "-prj", post_synth_prj.name,
                "-L", "smartconnect_v1_0",
                "-L", "axi_protocol_checker_v1_1_12",
                "-L", "axi_protocol_checker_v1_1_13",
                "-L", "axis_protocol_checker_v1_1_11",
                "-L", "axis_protocol_checker_v1_1_12",
                "-L", "xil_defaultlib",
                "-L", "unisims_ver",
                "-L", "xpm",
                "-L", "floating_point_v7_1_21",
                "-L", "floating_point_v7_0_26",
                "--lib", "ieee_proposed=./ieee_proposed",
                "-L", "uvm",
                "-relax",
                "-i", "./svr",
                "-i", "./axivip",
                "-i", "./svtb",
                "-i", "./file_agent",
                "-i", "./snn_to_hls_subsystem",
                "-s", "post_synth_sim",
                "-debug", "all",
            ]
            xsim_args = [
                "xsim",
                "-testplusarg", "UVM_VERBOSITY=UVM_NONE",
                "-testplusarg", "UVM_TESTNAME=snn_to_hls_test_lib",
                "-testplusarg", "UVM_TIMEOUT=20000000000000",
                "--noieeewarnings",
                "post_synth_sim",
                "-tclbatch", post_synth_tcl.as_posix(),
                "-view", (sim_verilog_dir / "snn_to_hls_dataflow_ana.wcfg").resolve().as_posix(),
                "-protoinst", (sim_verilog_dir / "snn_to_hls.protoinst").resolve().as_posix(),
            ]

            xsim_saif_bat = (sim_verilog_dir / "run_xsim_saif.bat").resolve()
            xsim_saif_sh = (sim_verilog_dir / "run_xsim_saif.sh").resolve()

            if os.name == "nt":
                with open(xsim_saif_bat, "w", encoding="utf-8") as f:
                    f.write(f'@echo off\r\n')
                    f.write(f'cd /d "%~dp0"\r\n')
                    f.write("call C:/AMDDesignTools/2025.2/Vivado/bin/" + " ".join(f'"{arg}"' if " " in str(arg) else str(arg) for arg in xelab_args) + "\r\n")
                    f.write("call C:/AMDDesignTools/2025.2/Vivado/bin/" + " ".join(f'"{arg}"' if " " in str(arg) else str(arg) for arg in xsim_args) + "\r\n")
                self._run_vitis_command([str(xsim_saif_bat)])
            else:
                self._write_shell_script(xsim_saif_sh, [xelab_args, xsim_args], cwd=sim_verilog_dir)
                self._run_vitis_command(["bash", str(xsim_saif_sh)])

        except Exception as e:
            print(f"\n*** Unable to build post-synthesis SAIF artifacts: {e}")

    def generate_power_report(self, solution_name="sol", post_synth=True):
        print(f"\n*** [Step 4] Generating Power Report from SAIF (post_synth={post_synth}) ***")
        self._generate_power_report_from_saif(solution_name, post_synth=post_synth)

    def run_power_report_from_saif(self, solution_name = "sol"):
        self._generate_power_report_from_saif(solution_name)

    def _generate_power_report_from_saif(self, solution_name = "sol", post_synth=False):

        try:
            base_dir = Path(self._folder_path) / self._project_name / solution_name
            impl_verilog_dir = base_dir / "impl" / "verilog"
            vivado_checkpoint = (impl_verilog_dir / "post_synth.dcp").resolve()
            if post_synth:
                saif_file = (impl_verilog_dir / "post_synth_activity.saif").resolve()
            else:
                saif_file = (impl_verilog_dir / "cosim_activity.saif").resolve()
            power_report = (impl_verilog_dir / "power_report_saif.txt").resolve()

            if not vivado_checkpoint.is_file():
                raise FileNotFoundError(f"Missing synthesis checkpoint: {vivado_checkpoint}")

            if not saif_file.is_file() or saif_file.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty SAIF file: {saif_file}")

            # Keep SAIF hierarchy aligned with the wrapper + DUT instance.
            strip_path = "apatb_snn_to_hls_top/AESL_inst_snn_to_hls"
            
            candidate_out_file = (impl_verilog_dir / f"power_report_saif_unmatched.txt").resolve()
            candidate_script = (impl_verilog_dir / f"run_power_report_from_saif.tcl").resolve()

            with open(candidate_script, "w", encoding="utf-8") as f:
                f.write("create_project -in_memory -part xcu250-figd2104-2L-e\n")
                f.write(f"open_checkpoint {vivado_checkpoint.as_posix()}\n")
                if strip_path is None:
                    f.write(f"read_saif -no_strip -out_file {candidate_out_file.as_posix()} {saif_file.as_posix()}\n")
                else:
                    f.write(f"read_saif -strip_path {strip_path} -out_file {candidate_out_file.as_posix()} {saif_file.as_posix()}\n")
                f.write(f"report_power -file {power_report.as_posix()} -name {{power_1}}\n")
                f.write("close_project\n")

            self._run_vitis_command(["vivado", "-mode", "batch", "-source", str(candidate_script)])

            if power_report.is_file():
                match_pattern = re.compile(r"Design Nets Matched\s*\|\s*([0-9.]+)%\s*\((\d+)/(\d+)\)")
                with open(power_report, "r", encoding="utf-8", errors="ignore") as report_handle:
                    for line in report_handle:
                        match = match_pattern.search(line)
                        if match:
                            print(f"\n*** SAIF Map Match: {match.group(1)}% ({match.group(2)}/{match.group(3)} nets)")
                            break
            
            if candidate_out_file.is_file():
                print(f"*** Unmatched net details saved at: {candidate_out_file}")
            
        except Exception as e:
            print(f"\n*** Unable to run power report from SAIF: {e}")

    def run_power_report_with_saif(self, solution_name = "sol"):
        self.configure_cosim_and_export(solution_name)
        self.run_vivado_synthesis(solution_name)
        self.generate_post_synth_saif(solution_name)
        self.generate_power_report(solution_name, post_synth=True)

    def get_synth_resource_usage(self, solution_name = "sol"):

        import xml.etree.ElementTree as ET

        csynth_file = Path(self._folder_path) / self._project_name / solution_name / "syn" / "report" / "snn_to_hls_csynth.xml"

        if not csynth_file.is_file():
            print("ERROR: Synth report file does not exist. Try running 'run_synth' before.")
            return

        tree = ET.parse(csynth_file)
        root = tree.getroot()

        area = root.find("AreaEstimates")
        resources = area.find("Resources")

        resource_usage = {
            "BRAM_18K": int(resources.find("BRAM_18K").text),
            "DSP": int(resources.find("DSP").text),
            "FF": int(resources.find("FF").text),
            "LUT": int(resources.find("LUT").text),
            "URAM": int(resources.find("URAM").text),
        }

        return resource_usage
    
    def get_synth_performance_estimates(self, solution_name = "sol"):

        import xml.etree.ElementTree as ET

        csynth_file = Path(self._folder_path) / self._project_name / solution_name / "syn" / "report" / "snn_to_hls_csynth.xml"

        if not csynth_file.is_file():
            print("ERROR: Synth report file does not exist. Try running 'run_synth' before.")
            return

        tree = ET.parse(csynth_file)
        root = tree.getroot()
        
        performance = root.find("PerformanceEstimates")
        latency = performance.find("SummaryOfOverallLatency")

        performance_estimates = {
            "avg_total_cycles": int(latency.find("Average-caseLatency").text),
            "avg_latency": latency.find("Average-caseRealTimeLatency").text
        }

        return performance_estimates
        
