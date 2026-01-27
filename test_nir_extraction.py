import nir
from NeuroHls.ModelConfigTobias import create_model_config_from_nir

# Carregar o arquivo NIR
nir_path = 'braille_noDelay_bias_zero.nir'
print(f"Carregando arquivo NIR: {nir_path}")
nir_graph = nir.read(nir_path)

print("\n=== Informações do Grafo NIR ===")
print(f"Número de nós: {len(nir_graph.nodes)}")
print(f"Número de edges: {len(nir_graph.edges)}")

print("\n=== Nós do Grafo ===")
for node_name, node in nir_graph.nodes.items():
    print(f"{node_name}: {type(node).__name__}")

print("\n=== Edges do Grafo ===")
for edge in nir_graph.edges:
    print(f"{edge[0]} -> {edge[1]}")

# Criar ModelConfig a partir do NIR
print("\n=== Criando ModelConfig a partir do NIR ===")
try:
    model_config = create_model_config_from_nir(nir_graph)
    print(f"✓ ModelConfig criado com sucesso!")
    print(f"  Número de camadas: {len(model_config.layers)}")
    
    print("\n=== Camadas Extraídas ===")
    for i, layer in enumerate(model_config.layers):
        print(f"{i}: {type(layer).__name__} - {layer}")
        
except Exception as e:
    print(f"✗ Erro ao criar ModelConfig: {e}")
    import traceback
    traceback.print_exc()
