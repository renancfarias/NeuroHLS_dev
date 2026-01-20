import nir
import numpy as np
import os
import re
import sys
from typing import Dict, List, Tuple, Any, Iterable

from collections import defaultdict
from collections import deque

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

    print(f"graph: {graph}")
    print(f"dep: {dependencies}\n\n")

    visited = set()
    queue = deque()

    queue.append("input")
    while queue:
        cur = queue.popleft()
        visited.add(cur)

        print(f"{cur}:")

        print("  - gets: ", end="")
        for dep in dependencies[cur]:
            if dep not in visited:
                print(f"{dep} (recurrent), ", end="")
            else:
                print(f"{dep} (ready), ", end="")
        print()

        print("  - sends: ", end="")
        for node in graph[cur]:
            if node not in visited:
                queue.append(node)
                print(f"{node}, ", end="")
            else:
                print(f"{node} (back), ", end="")
            
        print("\n\n")

# teste("z_nir_examples/lif_norse.nir")
teste("z_nir_examples/braille_noDelay_bias_zero.nir")