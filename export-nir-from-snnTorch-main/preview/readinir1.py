import nir

nir_graph = nir.read("csnn_mnist.nir")

nodes = nir_graph.nodes # A Dictionary of str -> nir.NIRNode
edges = nir_graph.edges # A List tuples (str, str)

for name, node in nodes.items():
    print(name, node)
    