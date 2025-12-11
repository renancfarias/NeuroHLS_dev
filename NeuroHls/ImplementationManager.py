from pathlib import Path
import shutil

from NeuroHls.ModelConfig import *
from NeuroHls.FileGenUtils import *
from NeuroHls.CodeCreator import *

class ImplementationManager:

    def __init__(self, folder_path: str):

        self._folder_path = folder_path
        self._AVAILABLE_ACTIVATIONS = {"LIF"} # Usar Json?

    def create_files_from_config(self, model_config: ModelConfig):

        net_input_shape = model_config.layers[0].get_input_shape()
        net_output_shape = model_config.layers[-1].get_output_shape()

        self._impl_code = CodeCreator(self._folder_path)

        self._create_network_header_and_prototype(net_input_shape, net_output_shape)
        
        self._cur_layer = 1
        self._last_output_name = "input"
        self._define_new_expected_input_shape(net_input_shape)
    
        self.used_types = {"input_t" : (model_config.input_total_bits, model_config.input_int_bits)}
        self.has_defined_output_layer = False

    def _create_network_header_and_prototype(self, input_shape, output_shape):

        bracket_input_shape = get_bracket_notation_of_tuple(input_shape)
        bracket_output_shape = get_bracket_notation_of_tuple(output_shape)

        self._impl_code.add_include("types_and_params.h")
        self._impl_code.add_include("neuro_hls_functions/bit_type.h")
        self._impl_code.add_include("neuro_hls_functions/dense.h")

        self._impl_code.add_code(f"void snn_to_hls(input_t input{bracket_input_shape}, bit_t output{bracket_output_shape})\n{'{'}\n")

        self._prototype = f"void snn_to_hls(input_t input{bracket_input_shape}, bit_t output{bracket_output_shape});\n"

    def _print_cur_layer(self):
        num_dashes = 50

        self._impl_code.add_code("\n//" + num_dashes * "-" + "\n")
        self._impl_code.add_code(f"//\tLayer {self._cur_layer}\n")
        self._impl_code.add_code("//" + num_dashes * "-" + "\n\n")

    def _check_input_shape(self, input_shape):
        
        if input_shape != self._expected_input_shape:
            raise Exception(f"input shape {input_shape} != expected {self._expected_input_shape}")

    def _define_new_expected_input_shape(self, new_input_shape):

        self._expected_input_shape = new_input_shape

    def _get_activation_function(self, activation):

        activation = activation.upper()

        if activation not in self._AVAILABLE_ACTIVATIONS:
            raise Exception(f"Activation {activation} not implemented")
        
        return activation
    
    def _prepare_for_next_layer(self):
        self._last_output_name = f"spikes_{self._cur_layer}"
        self._cur_layer += 1

    def _append_line(self, line):

        self._impl_code.add_code(f"\t{line}\n")

    def _check_output_shape(self, output_shape):
        
        if len(output_shape) != 1:
            raise Exception("No support for output shape with dim != 1")

    # def conv_2d(self, in_h : int, in_w : int, ker_h : int, ker_w : int, c_in : int, c_out : int, stride : int, result_type : str, is_output_layer = False, activation = "LIF"):
        
    #     if self.has_defined_output_layer:
    #         raise Exception("Output has already been defined. Cannot add any other layer")
        
    #     out_conv_h = (in_h - ker_h) // stride + 1
    #     out_conv_w = (in_w - ker_w) // stride + 1

    #     input_shape = (c_in, in_h, in_w)
    #     output_shape = (c_out, out_conv_h, out_conv_w)

    #     if is_output_layer:
    #         self._check_output_shape(output_shape)

    #     self._check_input_shape(input_shape)
    #     self._define_new_expected_input_shape(output_shape)

    #     output_shape = get_bracket_notation_of_tuple(output_shape)
    #     activation = self._get_activation_function(activation)

    #     self.used_types.add(result_type)
        
    #     self._print_cur_layer()

    #     potentials_var_name = f"potentials_{self._cur_layer}"
    #     spikes_var_name = "output" if is_output_layer else f"spikes_{self._cur_layer}"
    #     weight_var_name = f"weights_{self._cur_layer}"
    #     bias_var_name = f"bias_{self._cur_layer}"

    #     self._append_line(f"static {result_type} {potentials_var_name}{output_shape} = {{}};")

    #     if not is_output_layer:
    #         self._append_line(f"bit_t {spikes_var_name}{output_shape};\n")
    #     else:
    #         self._append_line("")

    #     self._append_line(f"conv_2d<{in_h}, {in_w}, {ker_h}, {ker_w}, {c_in}, {c_out}, {stride}>({self._last_output_name}, {potentials_var_name}, {weight_var_name}, {bias_var_name});")
    #     self._append_line(f"conv_2d_{activation}<{c_out}, {out_conv_h}, {out_conv_w}>({potentials_var_name}, {spikes_var_name});")

    #     if is_output_layer:
    #         self._finish_header(output_shape)
    #         self.has_defined_output_layer = True
    #     else:
    #         self._prepare_for_next_layer()

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

        self._append_line(f"static {result_type} {potentials_var_name}{output_shape} = {{}};")

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

    def _generate_types_file(self, path):

        types_file = CodeCreator(self._folder_path)

        types_file.add_include("ap_fixed.h")
        types_file.add_include("neuro_hls_functions/bit_type.h")

        # FINALIZAR PARA USAR DICT

        for type in self.used_types:
            types_and_params += f"typedef ap_fixed<16, 8> {type};\n"

        types_file.create_header_file("net_types")

    def _generate_header_file(self, folder_path):

        header = CodeCreator(folder_path)

        header.add_include("types_and_params.h")
        header.add_include("neuro_hls_functions/bit_type.h")
        header.add_include("neuro_hls_functions/dense.h")

        header.add_code(self._prototype)
        header.create_header()

    def generate_files(self, folder_path):

        if not self.has_defined_output_layer:
            raise Exception("Cannot generate files because output layer was not defined")

        self._cpp = self._impl_code + self._cpp
        self._cpp += "}\n"

        Path(folder_path).mkdir(parents=True, exist_ok=True)

        with open(f"{folder_path}/snn_implementation.cpp", "w") as f:
            f.write(self._cpp)

        # copy_folder_from_backend("neuro_hls_functions", folder_path)

        self._generate_types_file(folder_path)
        self._generate_header_file(folder_path)
