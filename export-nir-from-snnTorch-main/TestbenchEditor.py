from FileGenUtils import *

class TestbenchEditor:

    def __init__(self, input_component_count: int, output_size: int, step_count: int, different_sample_per_step: bool, folder_path):

        self._input_component_count = input_component_count
        self._output_size = output_size
        self._step_count = step_count
        self._different_sample_per_step = different_sample_per_step
        self._folder_path = folder_path

    def _get_decl_input_data(self):

        input_dims = ""

        if self.different_sample_per_step:
            input_dims += "[STEP_COUNT]"

        for i in range(1, self._input_component_count + 1):
            input_dims += f"[DIM_{i}]"

        return f"input_t input_data[BATCH_SIZE]{input_dims};"

    def finish_main(self):

        tb_name = f"{self._folder_path}/testbench.cpp"

        with open(tb_name, "r", encoding="utf-8") as f:
            tb_cpp = f.read()

        # Declaration input_data

        tb_cpp = tb_cpp.replace("//<decl_input_data>", self._get_decl_input_data())

        # Feed data to the SNN

        step_dimension_string = "[s]" if self.different_sample_per_step else ""
        feed_data_snn = f"snn_mnist_hls(input_data[b]{step_dimension_string}, output);"
        tb_cpp = tb_cpp.replace("//<feed_data_snn>", feed_data_snn)

        # Read data from input file

        input_file_read = "input_file >> input_data[b]"
        read_batch = IndentationMaker(3, first_line_should_use_indentation=False)

        idx = 1

        for dim in self.constants.keys():

            if dim == "OUTPUT_SIZE":
                continue

            if dim == "STEP_COUNT":
                
                if not self.different_sample_per_step:
                    continue

                read_batch.append_line(f"for (int s = 0; s < {dim}; s++)")
                input_file_read += "[s]"
            else:
                read_batch.append_line(f"for (int d{idx} = 0; d{idx} < {dim}; d{idx}++)")
                input_file_read += f"[d{idx}]"
                idx += 1

            read_batch.add_scope()

        read_batch.append_line(input_file_read + ";")
        tb_cpp = tb_cpp.replace("//<read_batch>", read_batch.get_text())

        with open(tb_name, "w", encoding="utf-8") as f:
            f.write(tb_cpp)