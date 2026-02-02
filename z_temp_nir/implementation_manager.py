from config_layers import *
from read_nir import read_nir

def get_bracket_str(arr):
    if arr is not None:
        return ''.join(f'[{x}]' for x in arr)

    return ""

def implement_model(model):

    # (id da camada do NIR, nome a ser usado na impl)
    layer_names = {}
    
    print(f"snn_to_hls(input_t input{get_bracket_str(model.input_shape)}, bit_t output{get_bracket_str(model.output_shape)})\n{'{'}", end="")

    for (idx, layer) in enumerate(model.layers[1:-1]):

        print(f"\n// implementation of '{layer.name}' layer\n")

        if isinstance(layer, Merge):

            for (dep_name, is_recurrent) in layer.dependencies:

                if is_recurrent and dep_name not in layer_names:
                    
                    impl_dep_name = f"layer_{len(layer_names) + 1}_rec"
                    layer_names[dep_name] = impl_dep_name

                    rec_layer_output_type = "bit_t" if layer.emits_spike else "type_t"
                    print(f"    {rec_layer_output_type} {impl_dep_name}{get_bracket_str(layer.input_shape)} = {{}};")

            print(f"    merge({layer_names.get(layer.layer_1)}, {layer_names.get(layer.layer_2)});")
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
                print(f"    {output_type} {name}{get_bracket_str(layer.output_shape)} = {{}};")
        else:
            name = layer_names[layer.name]

        # Chamando a funcao
        print(f"    {type(layer).__name__}({input_accum_name}, {name});")
    
    print("}")