import nir
from config_layers import *
from create_layer_config_from_node import *

from collections import defaultdict
from collections import deque

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

    model_config = ModelConfig()

    queue.append("input")
    while queue:
        
        cur = queue.popleft()
        visited.add(cur)

        # print(f"\n\ncurrent: {cur}")

        node_info = nodes[cur]
        func_name = type(node_info).__name__

        cur_layer = create_layer_config_from_node(node_info)

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

        # print(cur_layer)

        # if func_name == "Input":
        #     model_config.define_input(cur_layer)
        #     continue

        # if func_name == "Output":
        #     model_config.define_output(cur_layer)
        #     continue

        model_config.add_layer(cur_layer)
    
    return model_config

def teste(nir_file):
    model = read_nir(nir_file)
    print(model)

# teste("z_nir_examples/lif_norse.nir")
# teste("z_nir_examples/cnn_sinabs.nir")
teste("z_nir_examples/braille_noDelay_bias_zero.nir")