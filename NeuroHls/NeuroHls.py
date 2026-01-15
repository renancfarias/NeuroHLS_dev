import subprocess
import os
from pathlib import Path

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
        self._impl_manager = ImplementationManager(folder_path)

        self._project_name = "vitis_proj"

    def get_model_config_from_nir(self, nir):
        """
        Extrai a configuração do modelo a partir de um arquivo NIR.
        
        Args:
            nir: Caminho para o arquivo .nir ou objeto NIR
        
        Returns:
            ModelConfig: Configuração do modelo extraída do NIR
        """
        # Importa o parser NIR do mesmo pacote
        from .nir_to_c import NIRToCppParser
        
        # Se nir é uma string, assume que é o caminho do arquivo
        if isinstance(nir, str):
            parser = NIRToCppParser(nir)
        else:
            # Se não for string, pode ser que já seja o objeto nir_graph
            # Neste caso, cria um parser temporário
            raise ValueError("Por favor, forneça o caminho para o arquivo .nir como string")
        
        # Extrai o ModelConfig do parser
        model_config, input_shape, output_size = parser.get_model_config_from_nir()
        
        # Armazena os valores extraídos
        self._input_shape = input_shape
        self._output_size = output_size
        self._has_parsed_nir = True

        return model_config
    
    def get_dummy_model_config(self):
        
        model_config = ModelConfig()

        layer1 = DenseLayerConfig(784, 128)
        layer2 = DenseLayerConfig(128, 10)

        model_config.add_layer(layer1)
        model_config.add_layer(layer2)

        self._input_shape = (784,)
        self._output_size = 10
        self._has_parsed_nir = True

        return model_config
    
    def implement_model_from_config(self, model_config: ModelConfig):
        
        self._impl_manager.create_files_from_config(model_config)

    def define_test_dataset(self, npz_file: str, data_is_binary: bool, step_count: int, different_sample_per_step: bool):
        
        self._tb_manager.define_dataset(npz_file, data_is_binary, step_count, different_sample_per_step)

    def create_testbench(self, total_samples: int, batch_size: int):

        if not self._has_parsed_nir:
            print("ERROR: The network architecture must be defined before creating the testbench files.")
            return
        
        used_total_samples, used_batch_size = self._tb_manager.define_sample_count_and_batch_size(total_samples, batch_size)

        print(f"Total samples used: {used_total_samples} of {self._tb_manager.get_number_of_available_samples()}")
        print(f"Batch size: {used_batch_size}")
        print(f"Total batches: {used_total_samples // used_batch_size}")
        
        self._tb_manager.create_testbench_file(self._input_shape, self._output_size)
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

    def run_synth(self, clk_period_ms: int, part = "xc7z020clg400-1", solution_name = "sol"):

        try:
            self._create_vitis_project_if_needed()
            subprocess.run(["vitis_hls", "2_synth.tcl", self._project_name, solution_name, str(clk_period_ms), part], cwd = self._folder_path)
        except Exception:
            print("\n*** Unable to run Synthesis.")
        