import subprocess
import os

from .FileGenUtils import *
from .TestbenchManager import *

class NeuroHls:

    def __init__(self, folder_path: str, should_recreate_files = True):

        self._folder_path = folder_path

        if should_recreate_files or not os.path.exists(folder_path):
            copy_backend_to(folder_path)

        self._has_parsed_nir = False
        self._has_created_testbench = False

        self._tb_manager = TestbenchManager(folder_path)

    def parse_nir(self, nir):

        # Obter esses valores com o NIR
        self._input_shape = (784,)
        self._output_size = 10

        self._has_parsed_nir = True

    def create_test_dataset(self, dataloader, step_count: int, different_sample_per_step: bool):
        pass

    def define_testbench_parameters(self, total_samples: int, batch_size: int):
        pass

    def create_testbench(self, step_count: int, different_sample_per_step: bool):

        if not self._has_parsed_nir:
            print("ERROR: The network architecture must be defined before creating the testbench files.")
            return
        
        self._tb_manager.create_testbench_file(self._input_shape, self._output_size, step_count, different_sample_per_step)
        self._has_created_testbench = True

    def run_csim(self):

        if not self._has_created_testbench:
            print("ERROR: The testbench files must be created before running the C-Simulation.")
            return
        
        subprocess.run(["vitis_hls", "-f", "run_csim.tcl"], cwd = self._folder_path)

    def run_synth(self):
        pass