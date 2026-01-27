import nir
import numpy as np
import os
import re
import sys
from typing import Dict, List, Tuple, Any, Iterable

from collections import defaultdict
from collections import deque

def get_bracket_str(arr):
    if arr is not None:
        return ''.join(f'[{x}]' for x in arr)

    return ""

class LayerConf:
    def __init__(self, name, func_name, input_shape, output_shape):
        self.dependencies = []
        self.name = name
        self.func_name = func_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.is_recurrent = False

    def add_dependency(self, name: str, is_recurrent: bool):
        self.dependencies.append((name, is_recurrent))

    def __str__(self):
        s = "\n------------\n"
        s += f"layer: {self.name}\n"
        s += f"input {self.input_shape}\n"
        s += f"output: {self.output_shape}\n"
        s += f"is recurrent: {'yes' if self.is_recurrent else 'no'}\n"

        s += "dependencies:\n"
        for (name, is_recurrent) in self.dependencies:
            s += f"  - {name} ({'recurrent' if is_recurrent else 'ready'})\n"

        return s


class ModelConf:
    def __init__(self):
        self.layers = []
    
    def define_input(self, layer: LayerConf):
        self.layers.append(layer)
        self.input_shape = layer.input_shape

    def define_output(self, layer: LayerConf):
        self.layers.append(layer)
        self.output_shape = layer.output_shape

    def add_layer(self, layer: LayerConf):
        self.layers.append(layer)

    def __str__(self):
        s = ""

        for layer in self.layers:
            s += layer.__str__()

        return s

def read_nir(nir_file: str):

    print("\n" + "-" * 60)
    print(f"Abrindo {nir_file}")
    print("-" * 60 + "\n")

    nir_graph = nir.read(nir_file)
    nodes = nir_graph.nodes
    edges = nir_graph.edges

    dependencies = defaultdict(list)
    graph = defaultdict(list)

    print(f"Edges: {edges}\n")

    for e in edges:        
        graph[e[0]].append(e[1])
        dependencies[e[1]].append(e[0])

    visited = set()
    queue = deque()

    model_conf = ModelConf()

    queue.append("input")
    while queue:
        cur = queue.popleft()
        visited.add(cur)

        node_info = nodes[cur]
        func_name = type(node_info).__name__

        input_shape = node_info.input_type['input']
        output_shape = node_info.output_type['output']

        cur_layer = LayerConf(cur, func_name, input_shape, output_shape)

        for node in graph[cur]:
            if node not in visited:
                queue.append(node)

        ### Checking if layer is recurrent

        for node in graph[cur]:
            if node in visited:
                cur_layer.is_recurrent = True

        ### Declaring dependencies (if recurrent)

        for dep in dependencies[cur]:
            is_dep_recurrent = False

            if dep not in visited:
                is_dep_recurrent = True
            
            cur_layer.add_dependency(dep, is_dep_recurrent)

        if func_name == "Input":
            model_conf.define_input(cur_layer)
            continue

        if func_name == "Output":
            model_conf.define_output(cur_layer)
            continue

        model_conf.add_layer(cur_layer)
    
    return model_conf

def implement_model(model: ModelConf):

    # (id da camada do NIR, nome a ser usado na impl)
    layer_names = {}
    rec_count = 0
    
    print("Implementacao dummy:\n\n")
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

def teste(nir_file):
    model = read_nir(nir_file)
    implement_model(model)

# teste("z_nir_examples/lif_norse.nir")
# teste("z_nir_examples/cnn_sinabs.nir")
teste("z_nir_examples/braille_noDelay_bias_zero.nir")