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
        self._impl_manager = ImplementationManager(folder_path)

        self._has_defined_project_and_solution = False

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
    
    def implement_model_from_config(self, model_config: ModelConfig):
        
        self._impl_manager.create_files_from_config(model_config)

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

    def define_project_and_solution(self, project_name, solution_name):

        self._project_name = project_name
        self._solution_name = solution_name

        self._has_defined_project_and_solution = True


    # As funcoes abaixo estao improvisadas para testar os novos arquivos TCL

    def create_project(self):

        subprocess.run(["vitis_hls", "0_create_project.tcl", self._project_name], cwd = self._folder_path)

    def run_csim(self):

        # if not self._has_defined_project_and_solution:
        #     print("ERROR: The project's and solution's name must be defined before running the C-Simulation")
        #     return
        
        # if not self._has_created_testbench:
        #     print("ERROR: The testbench files must be created before running the C-Simulation.")
        #     return
        
        subprocess.run(["vitis_hls", "1_csim.tcl", self._project_name, self._solution_name], cwd = self._folder_path)

    def run_synth(self, clk_period: int, part = "xc7z020clg400-1"):
        
        subprocess.run(["vitis_hls", "2_synth.tcl", self._project_name, self._solution_name, str(clk_period), part], cwd = self._folder_path)