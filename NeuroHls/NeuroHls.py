import subprocess
from .FileGenUtils import *

class NeuroHls:

    def __init__(self, folder_path: str, should_create_files = True):

        self._folder_path = folder_path

        if should_create_files:
            copy_backend_to(folder_path)

        self._has_parsed_nir = False
        self._has_created_testbench = False

    def parse_nir(self, nir):

        # Obter esses valores com o NIR
        self._input_shape = (784,)
        self._output_size = 10

        self._has_parsed_nir = True

    def create_test_dataset(self):
        pass

    def define_testbench_parameters(self):
        pass

    def create_testbench(self, step_count: int, different_sample_per_step: bool):

        if not self._has_parsed_nir:
            print("ERROR: The network architecture must be defined before creating the testbench files.")
            return
        
        self._has_created_testbench = True

    def run_csim(self):

        if not self._has_created_testbench:
            print("ERROR: The testbench files must be created before running the C-Simulation.")
            return
        
        subprocess.run(["vitis_hls", "-f", "run.tcl"], cwd = self._folder_path)

    def run_synth(self):
        pass