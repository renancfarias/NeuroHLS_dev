import os
import numpy as np

from .FileGenUtils import *

class TestbenchManager:

    def __init__(self, folder_path: str):

        self._folder_path = folder_path
        self._has_defined_dataset = False

        self._constants = {}

    def _build_constants_dict(self):

        # self._constants = {}

        for idx, dim in enumerate(self._input_shape, start=1):
            self._constants[f"DIM_{idx}"] = dim

        self._constants["OUTPUT_SIZE"] = self._output_size
        self._constants["STEP_COUNT"] = self._step_count

    def _get_decl_constants_code(self):

        self._build_constants_dict()

        constants_str = ""

        for (constant, constant_value) in self._constants.items():
            constants_str += f"#define {constant} {constant_value}\n"

        return constants_str

    def _get_decl_input_data_code(self):

        input_dims = ""

        if self._different_sample_per_step:
            input_dims += "[STEP_COUNT]"

        for i in range(1, len(self._input_shape) + 1):
            input_dims += f"[DIM_{i}]"

        return f"input_t input_data[BATCH_SIZE]{input_dims};"
    
    def _get_feed_snn_code(self):

        step_dimension_string = "[s]" if self._different_sample_per_step else ""
        return f"snn_to_hls(input_data[b]{step_dimension_string}, output);"
    
    def _get_read_batch_code(self):

        input_file_read = "input_file >> input_data[b]"
        read_batch = IndentationMaker(3, first_line_should_use_indentation=False)

        if self._different_sample_per_step:
            input_file_read += "[s]"
            
            read_batch.append_line(f"for (int s = 0; s < STEP_COUNT; s++)")
            read_batch.add_scope()
        
        for i in range(1, len(self._input_shape) + 1):
            input_file_read += f"[d{i}]"

            read_batch.append_line(f"for (int d{i} = 0; d{i} < DIM_{i}; d{i}++)")
            read_batch.add_scope()

        read_batch.append_line(input_file_read + ";")
        return read_batch.get_text()
    
    def define_dataset(self, dataloader, step_count: int, different_sample_per_step: bool):

        total_samples = 0
        total_labels = 0

        for batch_x, batch_y in dataloader:

            # FALTA CRIAR ARQUIVO TXT
            
            if hasattr(batch_x, "numpy"):
                batch_x = batch_x.numpy()
                batch_y = batch_y.numpy()

                total_samples += batch_x.shape[0]
                total_labels += batch_y.shape[0]

        if different_sample_per_step and total_samples != total_labels * step_count:

            raise Exception(f"Number of samples does not match. {total_samples} != {total_labels} * {step_count}")
        
        if not different_sample_per_step and total_samples != total_labels:

            raise Exception(f"Number of samples does not match. {total_samples} != {total_labels}")
        
        self._available_samples = total_samples
        self._step_count = step_count
        self._different_sample_per_step = different_sample_per_step

        self._has_defined_dataset = True

    def get_number_of_available_samples(self):

        return self._available_samples
    
    def define_sample_count_and_batch_size(self, total_samples: int, batch_size: int):

        total_samples = abs(total_samples)
        batch_size = max(abs(batch_size), 1)
        
        total_samples = min(total_samples, self._available_samples)
        batch_size = min(batch_size, total_samples)

        # Assures that batch_size divides total_samples
        batch_size = get_closest_divisor(total_samples, batch_size)

        self._constants["TOTAL_SAMPLES"] = total_samples
        self._constants["BATCH_SIZE"] = batch_size

        return (total_samples, batch_size)

    def create_testbench_file(self, input_shape: tuple, output_size: int):

        self._input_shape = input_shape
        self._output_size = output_size

        # copy_file_from_backend("testbench.cpp", self._folder_path)

        tb_name = f"{self._folder_path}/testbench.cpp"

        with open(tb_name, "r", encoding="utf-8") as f:
            tb_cpp = f.read()

        # Declaration input_data
        tb_cpp = tb_cpp.replace("//<decl_input_data>", self._get_decl_input_data_code())

        # Feed data to the SNN
        tb_cpp = tb_cpp.replace("//<feed_data_snn>", self._get_feed_snn_code())

        # Read data from input file
        tb_cpp = tb_cpp.replace("//<read_batch>", self._get_read_batch_code())

        # Define constants, such as the number of steps, output size, etc
        tb_cpp = tb_cpp.replace("//<decl_constants>", self._get_decl_constants_code())

        with open(tb_name, "w", encoding="utf-8") as f:
            f.write(tb_cpp)

    def get_status(self):

        test_dataset_path = f"{self._folder_path}/tb_data/data.txt"
        test_targets_path = f"{self._folder_path}/tb_data/targets.txt"

        status_dataset = "OK" if os.path.exists(test_dataset_path) else "MISSING"
        status_targets = "OK" if os.path.exists(test_targets_path) else "MISSING"

        print("\nTestbench Status:")

        print(f" - Test dataset: {status_dataset}")
        print(f" - Test targets: {status_targets}")

        overall_status = "YES" if status_dataset == "OK" and status_targets == "OK" else "NO"
        print(f"\nReady? {overall_status}")

# def test_testbench_manager():

#     tb_manager = TestbenchManager("tcl_test")

#     # tb_manager.define_dataset("blabla")
#     # tb_manager.define_sample_count_and_batch_size(100, 1)

#     tb_manager.create_testbench_file((784,), 10, 10, True)

#     tb_manager.get_status()

# test_testbench_manager()
