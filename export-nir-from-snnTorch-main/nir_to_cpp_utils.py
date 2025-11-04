
class GetCpp:

    _cur_layer = 1

    def conv_2d(in_h : int, in_w : int, ker_h : int, ker_w : int, c_in : int, c_out : int, stride : int, result_type : str, input_var_name = ""):

        if (input_var_name == ""):
            input_var_name = f"result_{GetCpp._cur_layer - 1}"

        result_var_name = f"result_{GetCpp._cur_layer}"

        # Declaration of result var

        out_conv_h = (in_h - ker_h) // stride + 1
        out_conv_w = (in_w - ker_w) // stride + 1

        result_str = f"{result_type} {result_var_name}[{c_out}][{out_conv_h}][{out_conv_w}];\n"

        # Call of conv_2d

        result_str += f"conv_2d<{in_h}, {in_w}, {ker_h}, {ker_w}, {c_in}, {c_out}, {stride}>({input_var_name}, {result_var_name});\n"
        return result_str
    
    def inc_cur_layer():
        GetCpp._cur_layer += 1

test_cpp = GetCpp.conv_2d(32, 32, 3, 3, 1, 16, 1, "int", input_var_name="input")
GetCpp.inc_cur_layer()
test_cpp += GetCpp.conv_2d(30, 30, 3, 3, 16, 32, 1, "int")

print(f"\n\n{test_cpp}")