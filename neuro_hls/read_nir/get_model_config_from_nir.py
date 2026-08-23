import nir
import json
from pathlib import Path
from .layer_configuration import *
from .create_layer_config_from_node import create_layer_config_from_node

from collections import defaultdict
from collections import deque

METADATA_TARGET_KEYS = ("model_file", "nir_file", "file", "path")
METADATA_STRUCTURE_KEYS = set(METADATA_TARGET_KEYS) | {"layers", "nodes", "node_metadata"}

def _load_json_metadata(metadata_file_path: Path):

    with open(metadata_file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _as_list(value):

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return value

    return [value]

def _metadata_matches_nir(metadata, metadata_file_path: Path, nir_file_path: Path):

    for key in METADATA_TARGET_KEYS:
        for target in _as_list(metadata.get(key)):
            target_path = Path(str(target))

            if target_path.name == nir_file_path.name:
                return True

            if not target_path.is_absolute():
                target_path = metadata_file_path.parent / target_path

            try:
                if target_path.resolve() == nir_file_path.resolve():
                    return True
            except FileNotFoundError:
                continue

    return False

def _find_metadata_file(nir_file_path: Path):

    same_stem_metadata = nir_file_path.with_name(f"{nir_file_path.stem}.metadata.json")

    if same_stem_metadata.is_file():
        return same_stem_metadata

    for metadata_file_path in sorted(nir_file_path.parent.glob("*.metadata.json")):
        metadata = _load_json_metadata(metadata_file_path)

        if _metadata_matches_nir(metadata, metadata_file_path, nir_file_path):
            return metadata_file_path

    return None

def _load_model_metadata(nir_file_path: str, metadata_file_path: str = None):

    nir_path = Path(nir_file_path)

    if metadata_file_path is not None:
        metadata_path = Path(metadata_file_path)
    else:
        metadata_path = _find_metadata_file(nir_path)

    if metadata_path is None:
        return {}

    return _load_json_metadata(metadata_path)

def _get_external_node_metadata(node_name: str, model_metadata):

    if not model_metadata:
        return {}

    external_metadata = {
        key: value
        for key, value in model_metadata.items()
        if key not in METADATA_STRUCTURE_KEYS
    }

    layer_metadata = (
        model_metadata.get("layers")
        or model_metadata.get("nodes")
        or model_metadata.get("node_metadata")
        or {}
    )

    if node_name in layer_metadata:
        external_metadata.update(layer_metadata[node_name])

    return external_metadata

def _apply_external_metadata(node_name: str, node_info, model_metadata):

    external_metadata = _get_external_node_metadata(node_name, model_metadata)

    if not external_metadata:
        return node_info

    node_metadata = dict(getattr(node_info, "metadata", {}) or {})
    node_metadata.update(external_metadata)
    node_info.metadata = node_metadata

    return node_info

def get_node_info(graph, dependencies, nodes, cur_node):
    
    node_info = nodes[cur_node]

    input_shape = node_info.input_type['input']
    output_shape = node_info.output_type['output']

    # NIR represents a scalar Scale with shape (1,), but its scalar is allowed
    # to broadcast over the statically known tensor supplied by the graph.
    if type(node_info).__name__ == "Scale" and np.asarray(node_info.scale).size == 1:
        adjacent_shape = None
        for dep in dependencies[cur_node]:
            adjacent_shape = nodes[dep].output_type['output']
            if adjacent_shape is not None:
                break
        if adjacent_shape is None:
            for next_node in graph[cur_node]:
                adjacent_shape = nodes[next_node].input_type['input']
                if adjacent_shape is not None:
                    break
        if adjacent_shape is not None:
            input_shape = adjacent_shape
            output_shape = adjacent_shape

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

def get_model_config_from_nir(nir_file_path: str, metadata_file_path: str = None):

    nir_graph = nir.read(nir_file_path)
    model_metadata = _load_model_metadata(nir_file_path, metadata_file_path)
    nodes = nir_graph.nodes
    edges = nir_graph.edges

    dependencies = defaultdict(list)
    graph = defaultdict(list)

    for e in edges:        
        graph[e[0]].append(e[1])
        dependencies[e[1]].append(e[0])

    visited = set()
    discovered = {"input"}
    queue = deque()

    model_config = ModelConfig()

    queue.append("input")
    while queue:
        
        cur = queue.popleft()
        visited.add(cur)

        node_info = get_node_info(graph, dependencies, nodes, cur)
        node_info = _apply_external_metadata(cur, node_info, model_metadata)
        cur_layer = create_layer_config_from_node(cur, node_info)
        model_config.graph_layers[cur] = cur_layer

        for node in graph[cur]:
            if node not in discovered:
                discovered.add(node)
                queue.append(node)

        ### Checking if layer is recurrent

        for node in graph[cur]:
            if node in visited:
                cur_layer.is_recurrent = True

        ### Adding dependencies and checking if they are recurrent

        for dep in dependencies[cur]:

            is_dep_recurrent = dep not in visited
            cur_layer.add_dependency(dep, is_dep_recurrent)

        model_config.add_layer(cur_layer)

    model_config.graph_edges = list(edges)
    return model_config
