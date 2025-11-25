from pathlib import Path
import shutil
from FileGenUtils import *

class GetCpp:

    def __init__(self, input_shape: tuple):
        self._cur_layer = 1
        self._define_new_expected_input_shape(input_shape)
        self._last_output_name = "input"

        input_type = "input_t"

        self.implemented_activations = {"LIF"}
        self.used_types = {input_type}
        self.has_defined_output_layer = False

        if not isinstance(input_shape, tuple):
            input_shape = (input_shape, )

        self.input_shape = input_shape

        bracket_input_shape = get_bracket_notation_of_tuple(input_shape)

        self._header = "\n#include \"types_and_params.h\"\n\n"

        self._header += "#include \"neuro_hls_functions/bit_type.h\"\n"
        self._header += "#include \"neuro_hls_functions/dense.h\"\n"
        
        self._header += f"\nvoid snn_to_hls({input_type} input{bracket_input_shape}, bit_t output"
        self._cpp = ""

    def _print_cur_layer(self):
        num_dashes = 50

        self._cpp += "\n//" + num_dashes * "-" + "\n"
        self._cpp += f"//\tLayer {self._cur_layer}\n"
        self._cpp += "//" + num_dashes * "-" + "\n\n"

    def _check_input_shape(self, input_shape):
        
        if input_shape != self._expected_input_shape:
            raise Exception(f"input shape {input_shape} != expected {self._expected_input_shape}")

    def _define_new_expected_input_shape(self, new_input_shape):

        if not isinstance(new_input_shape, tuple):
            new_input_shape = (new_input_shape, )

        self._expected_input_shape = new_input_shape

    def _get_activation_function(self, activation):

        activation = activation.upper()

        if activation not in self.implemented_activations:
            raise Exception(f"Activation {activation} not implemented")
        
        return activation
    
    def _prepare_for_next_layer(self):
        self._last_output_name = f"spikes_{self._cur_layer}"
        self._cur_layer += 1

    def _append_line(self, line):
        self._cpp += "\t" + line + "\n"

    def _finish_header(self, output_shape):

        if not isinstance(output_shape, str):
            output_shape = get_bracket_notation_of_tuple(output_shape)
        
        self._header += output_shape + ")\n{\n"

    def _check_output_shape(self, output_shape):
        
        if len(output_shape) != 1:
            raise Exception("No support for output shape with dim != 1")

    def conv_2d(self, in_h : int, in_w : int, ker_h : int, ker_w : int, c_in : int, c_out : int, stride : int, result_type : str, is_output_layer = False, activation = "LIF"):
        
        if self.has_defined_output_layer:
            raise Exception("Output has already been defined. Cannot add any other layer")
        
        out_conv_h = (in_h - ker_h) // stride + 1
        out_conv_w = (in_w - ker_w) // stride + 1

        input_shape = (c_in, in_h, in_w)
        output_shape = (c_out, out_conv_h, out_conv_w)

        if is_output_layer:
            self._check_output_shape(output_shape)

        self._check_input_shape(input_shape)
        self._define_new_expected_input_shape(output_shape)

        output_shape = get_bracket_notation_of_tuple(output_shape)
        activation = self._get_activation_function(activation)

        self.used_types.add(result_type)
        
        self._print_cur_layer()

        potentials_var_name = f"potentials_{self._cur_layer}"
        spikes_var_name = "output" if is_output_layer else f"spikes_{self._cur_layer}"
        weight_var_name = f"weights_{self._cur_layer}"
        bias_var_name = f"bias_{self._cur_layer}"

        self._append_line(f"{result_type} {potentials_var_name}{output_shape};")

        if not is_output_layer:
            self._append_line(f"bit_t {spikes_var_name}{output_shape};\n")
        else:
            self._append_line("")

        self._append_line(f"conv_2d<{in_h}, {in_w}, {ker_h}, {ker_w}, {c_in}, {c_out}, {stride}>({self._last_output_name}, {potentials_var_name}, {weight_var_name}, {bias_var_name});")
        self._append_line(f"conv_2d_{activation}<{c_out}, {out_conv_h}, {out_conv_w}>({potentials_var_name}, {spikes_var_name});")

        if is_output_layer:
            self._finish_header(output_shape)
            self.has_defined_output_layer = True
        else:
            self._prepare_for_next_layer()

    def dense(self, n_inputs : int, n_neurons : int, result_type : str, is_output_layer = False, activation = "LIF"):
        
        if self.has_defined_output_layer:
            raise Exception("Output has already been defined. Cannot add any other layer")
        
        input_shape = (n_inputs, )
        output_shape = (n_neurons, )

        if is_output_layer:
            self._check_output_shape(output_shape)

        self._check_input_shape(input_shape)
        self._define_new_expected_input_shape(output_shape)

        activation = self._get_activation_function(activation)
        self.used_types.add(result_type)

        output_shape = get_bracket_notation_of_tuple(output_shape)

        self._print_cur_layer()

        potentials_var_name = f"potentials_{self._cur_layer}"
        spikes_var_name = "output" if is_output_layer else f"spikes_{self._cur_layer}"
        weight_var_name = f"weights_{self._cur_layer}"
        bias_var_name = f"bias_{self._cur_layer}"

        self._append_line(f"{result_type} {potentials_var_name}{output_shape};")

        if not is_output_layer:
            self._append_line(f"bit_t {spikes_var_name}{output_shape};\n")
        else:
            self._append_line("")

        self._append_line(f"dense<{n_inputs}, {n_neurons}>({self._last_output_name}, {potentials_var_name}, {weight_var_name}, {bias_var_name});")
        self._append_line(f"dense_{activation}<{n_neurons}>({potentials_var_name}, {spikes_var_name});")

        if is_output_layer:
            self._finish_header(output_shape)
            self.has_defined_output_layer = True
        else:
            self._prepare_for_next_layer()

    def _generate_types_and_parameters_file(self, path):

        types_and_params = "#ifndef _TYPES_AND_PARAMS_H_\n#define _TYPES_AND_PARAMS_H_\n\n"
        types_and_params += "#include \"ap_fixed.h\"\n\n"

        types_and_params += "#include \"neuro_hls_functions/bit_type.h\"\n\n"

        for type in self.used_types:
            types_and_params += f"typedef ap_fixed<16, 8> {type};\n"

        types_and_params += "\n"
        types_and_params += "\n#endif"

        with open(f"{path}/types_and_params.h", "w") as f:
            f.write(types_and_params)
            
    def generate_files(self, folder_path):

        if not self.has_defined_output_layer:
            raise Exception("Cannot generate files because output layer was not defined")

        self._cpp = self._header + self._cpp
        self._cpp += "}\n"

        Path(folder_path).mkdir(parents=True, exist_ok=True)

        with open(f"{folder_path}/snn_implementation.cpp", "w") as f:
            f.write(self._cpp)

        copy_folder_from_backend("neuro_hls_functions", folder_path)

        self._generate_types_and_parameters_file(folder_path)

# def test_conv():

#     test_cpp = GetCpp((1, 32, 32))

#     test_cpp.conv_2d(32, 32, 3, 3, 1, 16, 1, "potential_t")
#     test_cpp.conv_2d(30, 30, 3, 3, 16, 32, 1, "potential_t", is_output_layer=True)

#     test_cpp.generate_files("gen_test")

def test_dense():
    test_cpp = GetCpp(784)

    test_cpp.dense(784, 128, "potential_t")
    test_cpp.dense(128, 10, "potential_t", is_output_layer=True)
    
    test_cpp.generate_files("gen_test")

# test_conv()
test_dense()
