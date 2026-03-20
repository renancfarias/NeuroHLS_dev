import subprocess
import os
from pathlib import Path

from .backend_utils import copy_backend_to
from .read_nir import get_model_config_from_nir
from .testbench_manager import TestbenchManager
from .implementation_manager import implement_model

class NeuroHls:

    def __init__(self, folder_path: str):

        self._folder_path = folder_path

        if not os.path.exists(folder_path):
            copy_backend_to(folder_path)

        self._has_parsed_nir = False
        self._has_created_testbench = False

        self._tb_manager = TestbenchManager(folder_path)

        self._project_name = "vitis_proj"

    def read_nir_file(self, nir_file_path):

        model = get_model_config_from_nir(nir_file_path)
        self._input_shape = model.input_shape
        self._output_shape = model.output_shape

        self._has_parsed_nir = True
        
        return model 

    def implement_model(self, model, use_float):

        implement_model(model, self._folder_path, use_float)

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

    def _create_vitis_project_if_needed(self):

        proj_path = Path(self._folder_path) / self._project_name

        if os.path.exists(proj_path):
            return
        
        print("Creating Vitis Project...\n")
        
        if not self._tb_manager.is_ready():
            print("\n*** Project creation aborted.")
            raise Exception("Missing file")

        subprocess.run(["vitis_hls", "0_create_project.tcl", self._project_name], cwd = self._folder_path)

    def run_csim(self, solution_name = "sol"):
        
        try:
            self._create_vitis_project_if_needed()
            subprocess.run(["vitis_hls", "1_csim.tcl", self._project_name, solution_name], cwd = self._folder_path)
        except Exception:
            print("\n*** Unable to run C-Simulation.")

    def run_synth(self, frequency_MHz: int, part = "xc7z020clg400-1", solution_name = "sol"):

        if frequency_MHz <= 0:
            print("ERROR: frequency must be greater than 0.\n")
            return

        clk_period_ns = 1000 / frequency_MHz

        try:
            self._create_vitis_project_if_needed()
            subprocess.run(["vitis_hls", "2_synth.tcl", self._project_name, solution_name, str(clk_period_ns), part], cwd = self._folder_path)
        except Exception:
            print("\n*** Unable to run Synthesis.")
        