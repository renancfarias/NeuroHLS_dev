import subprocess
import os

from .FileGenUtils import *
from .TestbenchManager import *
from .ModelConfig import *
from .ImplementationManager import *

class NeuroHls:

    def __init__(self, folder_path: str, should_recreate_files = True):

        self._folder_path = folder_path

        if should_recreate_files or not os.path.exists(folder_path):
            copy_backend_to(folder_path)

        self._has_parsed_nir = False
        self._has_created_testbench = False

        self._tb_manager = TestbenchManager(folder_path)

    def get_model_config_from_nir(self, nir):

        # Obter esses valores com o NIR

        self._input_shape = (784,)
        self._output_size = 10

        self._has_parsed_nir = True

        model_config = ModelConfig()
        model_config.add_layer(DenseLayerConfig(784, 128))
        model_config.add_layer(DenseLayerConfig(128, 10))

        return model_config
    
    def implement_model_from_config(self, model_config: ModelConfig):
        
        # REFATORAR IMPLEMENTATION_MANAGER

        impl_manager = ImplementationManager((784, ))

        for i in range(len(model_config.layers)):
            
            is_activation_layer = (i == len(model_config.layers) - 1)
            layer = model_config.layers[i]

            if isinstance(layer, DenseLayerConfig):
                impl_manager.dense(layer.n_inputs, layer.n_neurons, "potential_t", is_activation_layer)

        impl_manager.generate_files(self._folder_path)

    def create_test_dataset(self, dataloader, step_count: int, different_sample_per_step: bool):
        
        self._tb_manager.define_dataset(dataloader, step_count, different_sample_per_step)

    def define_testbench_parameters(self, total_samples: int, batch_size: int):
        
        used_total_samples, used_batch_size = self._tb_manager.define_sample_count_and_batch_size(total_samples, batch_size)

        print(f"Total samples used: {used_total_samples} of {self._tb_manager.get_number_of_available_samples()}")
        print(f"Batch size: {used_batch_size}")
        print(f"Total batches: {used_total_samples // used_batch_size}")

    def create_testbench(self):

        if not self._has_parsed_nir:
            print("ERROR: The network architecture must be defined before creating the testbench files.")
            return
        
        self._tb_manager.create_testbench_file(self._input_shape, self._output_size)
        self._has_created_testbench = True

        print("Testbench Criado")

    def run_csim(self):

        if not self._has_created_testbench:
            print("ERROR: The testbench files must be created before running the C-Simulation.")
            return
        
        subprocess.run(["vitis_hls", "-f", "run_csim.tcl"], cwd = self._folder_path)

    def run_synth(self):
        pass