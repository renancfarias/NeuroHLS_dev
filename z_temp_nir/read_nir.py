import nir
from config_layers import *
from create_layer_config_from_node import *

from collections import defaultdict
from collections import deque

def get_node_info(graph, dependencies, nodes, cur_node):
    
    node_info = nodes[cur_node]

    input_shape = node_info.input_type['input']
    output_shape = node_info.output_type['output']

    if input_shape is None:

        # input shape must be inferred

        for dep in dependencies[cur_node]:
            
            dep_info = nodes[dep]
            dep_output_shape = dep_info.output_type['output']

            if dep_output_shape is not None:
                input_shape = dep_output_shape
                break

    if output_shape is None:

        # output shape must be inferred

        for next_node in graph[cur_node]:

            next_node_info = nodes[next_node]
            next_node_input_shape = next_node_info.input_type['input']

            if next_node_input_shape is not None:
                output_shape = next_node_input_shape
                break

    if input_shape is None:
        raise Exception(f"Unable to infer the input shape of {cur_node} layer")
    
    if output_shape is None:
        raise Exception(f"Unable to infer the output shape of {cur_node} layer")
    
    node_info.input_type['input'] = input_shape
    node_info.output_type['output'] = output_shape

    return node_info

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

        node_info = get_node_info(graph, dependencies, nodes, cur)
        cur_layer = create_layer_config_from_node(cur, node_info)

        for node in graph[cur]:
            if node not in visited:
                queue.append(node)

        ### Checking if layer is recurrent

        for node in graph[cur]:
            if node in visited:
                cur_layer.is_recurrent = True

        ### Adding dependencies and checking if they are recurrent

        for dep in dependencies[cur]:
            is_dep_recurrent = False

            if dep not in visited:
                is_dep_recurrent = True
            
            cur_layer.add_dependency(dep, is_dep_recurrent)

        model_config.add_layer(cur_layer)
    
    return model_config

def teste(nir_file):
    model = read_nir(nir_file)
    print(model)

# teste("z_nir_examples/lif_norse.nir")
teste("z_nir_examples/cnn_sinabs.nir")
# teste("z_nir_examples/braille_noDelay_bias_zero.nir")