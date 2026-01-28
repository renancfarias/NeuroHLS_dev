from config_layers import *
from read_nir import read_nir

def get_bracket_str(arr):
    if arr is not None:
        return ''.join(f'[{x}]' for x in arr)

    return ""

def implement_model(model):

    # (id da camada do NIR, nome a ser usado na impl)
    layer_names = {}
    rec_count = 0
    
    print(f"snn_to_hls(input_t input{get_bracket_str(model.input_shape)}, bit_t output{get_bracket_str(model.output_shape)})\n{'{'}", end="")

    for (idx, layer) in enumerate(model.layers[1:-1]):

        print(f"\n// implementation of '{layer.name}' layer\n")

        # Declarando potenciais de camadas recorrentes que a camada atual usa

        for (dep_name, is_recurrent) in layer.dependencies:

            if is_recurrent and dep_name not in layer_names:
                
                rec_count += 1
                impl_dep_name = f"rec_{rec_count}"
                layer_names[dep_name] = impl_dep_name

                print(f"    type_t {impl_dep_name}{get_bracket_str(layer.input_shape)} = {{}};")

        # Dando merge nos inputs (caso tenha mais de um)

        input_accum_name = layer_names.get(layer.dependencies[0][0], layer.dependencies[0][0])
        for (dep_name, is_recurrent) in layer.dependencies[1:]:
            print(f"    merge({input_accum_name}, {layer_names[dep_name]});")

        # Declarando potencial da camada, caso ela nao seja recorrente

        if not layer.is_recurrent:

            if idx == len(model.layers) - 3:
                name = "output"
            else:
                name = f"layer_{len(layer_names) + 1}"
                layer_names[layer.name] = name
                print(f"    type_t {name}{get_bracket_str(layer.output_shape)} = {{}};")
        else:
            name = layer_names[layer.name]

        # Chamando a funcao
        print(f"    {layer.func_name}({input_accum_name}, {name});")
    
    print("}")