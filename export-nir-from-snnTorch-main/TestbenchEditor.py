from FileGenUtils import *

class TestbenchEditor:

    def __init__(self, input_component_count: int, output_size: int, step_count: int, different_sample_per_step: bool, folder_path):

        self._input_component_count = input_component_count
        self._output_size = output_size
        self._step_count = step_count
        self._different_sample_per_step = different_sample_per_step
        self._folder_path = folder_path

    def _get_decl_input_data_code(self):

        input_dims = ""

        if self._different_sample_per_step:
            input_dims += "[STEP_COUNT]"

        for i in range(1, self._input_component_count + 1):
            input_dims += f"[DIM_{i}]"

        return f"input_t input_data[BATCH_SIZE]{input_dims};"
    
    def _get_feed_snn_code(self):

        step_dimension_string = "[s]" if self._different_sample_per_step else ""
        return f"snn_mnist_hls(input_data[b]{step_dimension_string}, output);"
    
    def _get_read_batch_code(self):

        input_file_read = "input_file >> input_data[b]"
        read_batch = IndentationMaker(3, first_line_should_use_indentation=False)

        if self._different_sample_per_step:
            input_file_read += "[s]"
            
            read_batch.append_line(f"for (int s = 0; s < STEP_COUNT; s++)")
            read_batch.add_scope()
        
        for i in range(1, self._input_component_count + 1):
            input_file_read += f"[d{i}]"

            read_batch.append_line(f"for (int d{i} = 0; d{i} < DIM_{i}; d{i}++)")
            read_batch.add_scope()

        read_batch.append_line(input_file_read + ";")
        return read_batch.get_text()

    def finish_main(self):

        tb_name = f"{self._folder_path}/testbench.cpp"

        with open(tb_name, "r", encoding="utf-8") as f:
            tb_cpp = f.read()

        # Declaration input_data
        tb_cpp = tb_cpp.replace("//<decl_input_data>", self._get_decl_input_data_code())

        # Feed data to the SNN
        tb_cpp = tb_cpp.replace("//<feed_data_snn>", self._get_feed_snn_code())

        # Read data from input file
        tb_cpp = tb_cpp.replace("//<read_batch>", self._get_read_batch_code())

        with open(tb_name, "w", encoding="utf-8") as f:
            f.write(tb_cpp)

def test_testbench_editor():

    tb_editor = TestbenchEditor(1, 10, 10, False, "gen_test")
    tb_editor.finish_main()

test_testbench_editor()
