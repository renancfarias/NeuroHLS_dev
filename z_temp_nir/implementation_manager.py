from config_layers import *

NUM_DASHES_COMMENT = 50

def get_bracket_str(arr):
    if arr is not None:
        return ''.join(f'[{x}]' for x in arr)

    return ""

def implement_model(model):

    # (id da camada do NIR, nome a ser usado na impl)
    layer_names = {}

    code_header = f"void snn_to_hls(input_t input{get_bracket_str(model.input_shape)}, bit_t output{get_bracket_str(model.output_shape)})"
    code = f"{code_header}\n{'{'}"
    
    neuron_params_code = ""

    for (idx, layer) in enumerate(model.layers[1:-1]):

        code += "\n//" + "-" * NUM_DASHES_COMMENT
        code += f"\n// implementation of '{layer.name}' layer"
        code += "\n//" + "-" * NUM_DASHES_COMMENT + "\n\n"

        if isinstance(layer, Merge):

            for (dep_name, is_recurrent) in layer.dependencies:

                if is_recurrent and dep_name not in layer_names:
                    
                    impl_dep_name = f"layer_{len(layer_names) + 1}_rec"
                    layer_names[dep_name] = impl_dep_name

                    rec_layer_output_type = "bit_t" if layer.emits_spike else "type_t"
                    code += f"\t{rec_layer_output_type} {impl_dep_name}{get_bracket_str(layer.input_shape)} = {{}};\n"

            code += f"\tMerge({layer_names.get(layer.layer_1)}, {layer_names.get(layer.layer_2)});\n"
            continue

        input_accum_name = layer_names.get(layer.dependencies[0][0], layer.dependencies[0][0])

        # Declarando potencial da camada, caso ela nao seja recorrente

        if not layer.is_recurrent:

            if idx == len(model.layers) - 3:
                name = "output"
            else:
                name = f"layer_{len(layer_names) + 1}"
                layer_names[layer.name] = name
                output_type = "bit_t" if layer.emits_spike else "type_t"
                code += f"\t{output_type} {name}{get_bracket_str(layer.output_shape)} = {{}};\n"
        else:
            name = layer_names[layer.name]

        # Chamando a funcao
        code += f"\t{type(layer).__name__}({input_accum_name}, {name});\n"
    
    code += "}\n"

    print(code)