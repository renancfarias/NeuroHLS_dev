class GetCpp:

    def __init__(self, input_shape : tuple, output_shape : tuple):
        self._cur_layer = 1
        self._expected_input_shape = input_shape
        self._last_output_name = "input"

        self.implemented_activations = {"LIF"}

        input_shape = self._get_bracket_syntax_of_shape(input_shape)
        output_shape = self._get_bracket_syntax_of_shape(output_shape)

        self._cpp = f"void snn_to_hls(input{input_shape}, output{output_shape})\n"
        self._cpp += "{\n"

    def _get_bracket_syntax_of_shape(self, shape : tuple):

        if not isinstance(shape, tuple):
            shape = (shape, )

        brackets = ""

        for i in shape:
            brackets += f"[{i}]"
        
        return brackets

    def _print_cur_layer(self):
        self._cpp += "\n//---------------------\n"
        self._cpp += f"//\tLayer {self._cur_layer}\n"
        self._cpp += "//---------------------\n\n"

    def _check_input_shape(self, input_shape):
        
        if input_shape != self._expected_input_shape:
            raise Exception(f"input shape {input_shape} != expected {self._expected_input_shape}")

    def _define_new_expected_input_shape(self, new_input_shape):
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
        self._cpp += "\t" + line

    def conv_2d(self, in_h : int, in_w : int, ker_h : int, ker_w : int, c_in : int, c_out : int, stride : int, result_type : str, activation = "LIF"):

        out_conv_h = (in_h - ker_h) // stride + 1
        out_conv_w = (in_w - ker_w) // stride + 1

        input_shape = (c_in, in_h, in_w)
        output_shape = (c_out, out_conv_h, out_conv_w)

        self._check_input_shape(input_shape)
        self._define_new_expected_input_shape(output_shape)

        output_shape = self._get_bracket_syntax_of_shape(output_shape)
        activation = self._get_activation_function(activation)
        
        self._print_cur_layer()

        potentials_var_name = f"potentials_{self._cur_layer}"
        spikes_var_name = f"spikes_{self._cur_layer}"

        self._append_line(f"{result_type} {potentials_var_name}{output_shape};\n")
        self._append_line(f"bit_t {spikes_var_name}{output_shape};\n\n")

        self._append_line(f"conv_2d<{in_h}, {in_w}, {ker_h}, {ker_w}, {c_in}, {c_out}, {stride}>({self._last_output_name}, {potentials_var_name});\n")
        self._append_line(f"conv_2d_{activation}<{c_out}, {out_conv_h}, {out_conv_w}>({potentials_var_name}, {spikes_var_name});\n\n")

        self._prepare_for_next_layer()
        

    def dense(self, n_inputs : int, n_neurons : int, result_type : str, activation = "LIF"):
        
        self._check_input_shape((n_inputs))
        activation = self._get_activation_function(activation)

        self._print_cur_layer()

        potentials_var_name = f"potentials_{self._cur_layer}"
        spikes_var_name = f"spikes_{self._cur_layer}"

        ################
        # OBS: nesse codigo, os parametros do template da camada densa (n_inputs e n_neurons) estao invertidos
        ############

        self._cpp += f"{result_type} {potentials_var_name}[{n_neurons}];\n"
        self._cpp += f"bit_t {spikes_var_name}[{n_neurons}];\n\n"
        self._cpp += f"dense<{n_inputs}, {n_neurons}>({self._last_output_name}, {potentials_var_name});\n"
        self._cpp += f"dense_{activation}<{n_neurons}>({potentials_var_name}, {spikes_var_name});\n"

        self._prepare_for_next_layer()
        self._define_new_expected_input_shape((n_neurons))

    def get_cpp(self):
        self._cpp += "}\n"
        return self._cpp

def test_conv():

    test_cpp = GetCpp((1, 32, 32), (10))

    test_cpp.conv_2d(32, 32, 3, 3, 1, 16, 1, "potential_t")
    test_cpp.conv_2d(30, 30, 3, 3, 16, 32, 1, "potential_t")

    print("\n\n" + test_cpp.get_cpp())

def test_dense():
    test_cpp = GetCpp((784), (10))

    test_cpp.dense(784, 128, "potential_t")
    print("\n\n" + test_cpp.get_cpp())

test_conv()
# test_dense()