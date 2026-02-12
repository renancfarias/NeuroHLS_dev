from neuro_hls.read_nir import *
from .extract_neuron_param_code import extract_neuron_param_code
from .code_creator import CodeCreator

NUM_DASHES_COMMENT = 50

def get_bracket_str(arr):
    if arr is not None:
        return ''.join(f'[{x}]' for x in arr)

    return ""

def implement_model(model, folder_path):

    # (id da camada do NIR, nome a ser usado na impl)
    layer_names = {}

    model_prototype = f"void snn_to_hls(input_t input{get_bracket_str(model.input_shape)}, bit_t output{get_bracket_str(model.output_shape)})"

    model_cpp = CodeCreator(folder_path)
    model_h = CodeCreator(folder_path)
    neuron_params_h = CodeCreator(folder_path)
    quantization_h = CodeCreator(folder_path)

    model_h.add_code(f"{model_prototype};")
    model_cpp.add_code(f"{model_prototype}\n{'{'}")

    for (idx, layer) in enumerate(model.layers[1:-1]):

        model_cpp.add_code("\n//" + "-" * NUM_DASHES_COMMENT)
        model_cpp.add_code(f"\n// implementation of '{layer.name}' layer")
        model_cpp.add_code("\n//" + "-" * NUM_DASHES_COMMENT + "\n\n")

        if isinstance(layer, layer_configuration.Merge):

            for (dep_name, is_recurrent) in layer.dependencies:

                if is_recurrent and dep_name not in layer_names:
                    
                    impl_dep_name = f"layer_{len(layer_names) + 1}_rec"
                    layer_names[dep_name] = impl_dep_name

                    rec_layer_output_type = "bit_t" if layer.emits_spike else "type_t"
                    model_cpp.add_code(f"\t{rec_layer_output_type} {impl_dep_name}{get_bracket_str(layer.input_shape)} = {{}};\n")

            model_cpp.add_code(f"\tMerge({layer_names.get(layer.layer_1)}, {layer_names.get(layer.layer_2)});\n")
            continue

        if layer.name in layer_names:
            cur_layer_id = "_".join(layer_names[layer.name].split("_")[1:])
        else:
            cur_layer_id = len(layer_names) + 1

        neuron_params = layer.get_neuron_params()
        for name, value in neuron_params.items():
            neuron_params_h.add_code(f"weight_t {name}_{cur_layer_id}{get_bracket_str(value.shape)} = {extract_neuron_param_code(value)};\n\n")

        input_accum_name = layer_names.get(layer.dependencies[0][0], layer.dependencies[0][0])

        # Declarando potencial da camada, caso ela nao seja recorrente

        if not layer.is_recurrent:

            if idx == len(model.layers) - 3:
                name = "output"
            else:
                name = f"layer_{len(layer_names) + 1}"
                layer_names[layer.name] = name
                output_type = "bit_t" if layer.emits_spike else "potential_t"
                model_cpp.add_code(f"\t{output_type} {name}{get_bracket_str(layer.output_shape)} = {{}};\n")
        else:
            name = layer_names[layer.name]

        # Chamando a funcao
        neuron_params_call = ", ".join(f"{key}_{cur_layer_id}" for key in neuron_params.keys())
        
        if len(neuron_params_call) > 0: 
            neuron_params_call = ", " + neuron_params_call

        model_cpp.add_code(f"\t{type(layer).__name__}({input_accum_name}, {name}{neuron_params_call});\n")
    
    model_cpp.add_code("}\n")

    quantization_h.add_include("ap_int.h")
    quantization_h.add_code(f"typedef ap_int<{model.input_quantization[0]}, {model.input_quantization[1]}> input_t;\n")
    quantization_h.add_code(f"typedef ap_int<{model.weight_quantization[0]}, {model.weight_quantization[1]}, AP_RND> weight_t;\n")
    quantization_h.add_code(f"typedef ap_int<{model.potential_quantization[0]}, {model.potential_quantization[1]}> potential_t;\n")

    model_h.add_include("neuro_hls_functions/bit_type.h")
    model_h.add_include("quantization.h")

    model_cpp.add_include("neuro_hls_functions/bit_type.h")
    model_cpp.add_include("quantization.h")
    model_cpp.add_include("neuron_params.h")
    model_cpp.add_include("snn_implementation.h")

    neuron_params_h.add_include("quantization.h")
    
    model_h.create_file("snn_implementation.h")
    model_cpp.create_file("snn_implementation.cpp")
    neuron_params_h.create_file("neuron_params.h")
    quantization_h.create_file("quantization.h")
    
    