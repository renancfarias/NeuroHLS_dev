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

def teste(nir_file: str):

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

    queue.append("input")
    while queue:
        cur = queue.popleft()
        visited.add(cur)

        node_info = nodes[cur]
        func_name = type(node_info).__name__

        output_bracket = get_bracket_str(node_info.output_type['output'])

        for node in graph[cur]:
            if node not in visited:
                queue.append(node)

        if func_name == "Input":
            print(f"snn_hls(input{output_bracket})")
            continue

        ### Should declare potential (not recurrent)

        should_decl_potential = True
        for node in graph[cur]:
            if node in visited:
                should_decl_potential = False

        if should_decl_potential:
            print(f"    {cur}{output_bracket} = {{}}")

        ### Declaring dependencies (if recurrent)

        for dep in dependencies[cur]:
            if dep not in visited:
                print(f"    {dep}{get_bracket_str(nodes[dep].input_type['input'])} = {{}} // recurrent")

        ### Adding dependecies

        if len(dependencies[cur]) == 0:
            continue

        if len(dependencies[cur]) == 1: # Sem recorrencia
            print(f"    {func_name}({dependencies[cur][0]}, {cur});")
        else:
            for dep in dependencies[cur][1:]:
                print(f"    merge({dependencies[cur][0]}, {dep});")
            print(f"    {func_name}({dependencies[cur][0]}, {cur});")
        print()

# teste("z_nir_examples/lif_norse.nir")
# teste("z_nir_examples/cnn_sinabs.nir")
teste("z_nir_examples/braille_noDelay_bias_zero.nir")