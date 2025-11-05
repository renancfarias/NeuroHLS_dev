class GetCpp:

    def __init__(self, input_shape):
        self._cur_layer = 1
        self._expected_input_shape = input_shape
        self._last_output_name = "input"

        self.implemented_activations = {"LIF"}

        self.cpp = ""

    def _print_cur_layer(self):
        self.cpp += "\n//---------------------\n"
        self.cpp += f"//\tLayer {self._cur_layer}\n"
        self.cpp += "//---------------------\n\n"
        
    def conv_2d(self, in_h : int, in_w : int, ker_h : int, ker_w : int, c_in : int, c_out : int, stride : int, result_type : str, activation = "LIF"):

        cur_input_shape = (c_in, in_h, in_w)

        if cur_input_shape != self._expected_input_shape:
            raise Exception(f"input shape {cur_input_shape} != expected {self._expected_input_shape}")
        
        out_conv_h = (in_h - ker_h) // stride + 1
        out_conv_w = (in_w - ker_w) // stride + 1

        self._expected_input_shape = (c_out, out_conv_h, out_conv_w)

        activation = activation.upper()
        if activation not in self.implemented_activations:
            raise Exception(f"Activation {activation} not implemented")
        
        self._print_cur_layer()

        potentials_var_name = f"potentials_{self._cur_layer}"
        spikes_var_name = f"spikes_{self._cur_layer}"

        self.cpp += f"{result_type} {potentials_var_name}[{c_out}][{out_conv_h}][{out_conv_w}];\n"
        self.cpp += f"bit_t {spikes_var_name}[{c_out}][{out_conv_h}][{out_conv_w}];\n\n"
        self.cpp += f"conv_2d<{in_h}, {in_w}, {ker_h}, {ker_w}, {c_in}, {c_out}, {stride}>({self._last_output_name}, {potentials_var_name});\n"
        self.cpp += f"conv_2d_{activation}<{c_out}, {out_conv_h}, {out_conv_w}>({potentials_var_name}, {spikes_var_name});\n\n"

        self._last_output_name = spikes_var_name
        self._cur_layer += 1

test_cpp = GetCpp((1, 32, 32))

test_cpp.conv_2d(32, 32, 3, 3, 1, 16, 1, "potential_t")
test_cpp.conv_2d(30, 30, 3, 3, 16, 32, 1, "potential_t")

print(f"{test_cpp.cpp}")