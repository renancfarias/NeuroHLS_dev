#!/usr/bin/env python3
"""
Script de teste para verificar a função get_info_from_node e build_model_from_nir.
"""

from NeuroHls.ModelConfigTobias import build_model_from_nir, ModelConfig

def test_nir_file(nir_file: str):
    """
    Testa a leitura de um arquivo NIR e construção do ModelConfig.
    
    Args:
        nir_file: Caminho para o arquivo .nir
    """
    print("\n" + "=" * 80)
    print(f"Testando arquivo: {nir_file}")
    print("=" * 80 + "\n")
    
    try:
        # Constrói o modelo a partir do NIR
        model_config = build_model_from_nir(nir_file)
        
        # Imprime o modelo
        print("\n" + "-" * 80)
        print("MODELO CONFIGURADO:")
        print("-" * 80)
        print(model_config)
        print("-" * 80 + "\n")
        
        # Estatísticas
        print(f"Total de camadas: {len(model_config.layers)}")
        
        # Mostra os tipos de camadas
        print("\nTipos de camadas:")
        for idx, layer in enumerate(model_config.layers):
            layer_type = type(layer).__name__
            print(f"  {idx+1}. {layer_type}")
            print(f"     Input shape:  {layer.get_input_shape()}")
            print(f"     Output shape: {layer.get_output_shape()}")
        
        return model_config
        
    except Exception as e:
        print(f"❌ ERRO ao processar o arquivo: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    
    # Testa com o arquivo dense_only_linear1.nir se disponível
    if len(sys.argv) > 1:
        nir_file = sys.argv[1]
    else:
        # Arquivo padrão
        nir_file = "dense_only_linear1.nir"
    
    model = test_nir_file(nir_file)
    
    if model is not None:
        print("\n✅ Teste concluído com sucesso!")
    else:
        print("\n❌ Teste falhou!")
