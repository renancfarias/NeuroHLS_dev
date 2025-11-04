import nir

nir_graph = nir.read("csnn_mnist.nir")

nodes = nir_graph.nodes # A Dictionary of str -> nir.NIRNode
edges = nir_graph.edges # A List tuples (str, str)

print("=== NODES ===")
for name, node in nodes.items():
    print(f"{name}: {type(node).__name__}")
    
print("\n=== EDGES ===")
for edge in edges:
    print(f"{edge[0]} -> {edge[1]}")
