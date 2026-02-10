from neuro_hls.implementation_manager import implement_model
from neuro_hls.read_nir.get_model_config_from_nir import get_model_config_from_nir

def teste(nir_file):

    path = "z_temp_nir"

    model = get_model_config_from_nir(nir_file)
    print(model)

    # print("IMPLEMENTATION:\n\n")

    implement_model(model, path)


# teste("nir_examples/lif_norse.nir")
# teste("nir_examples/cnn_sinabs.nir")
teste("nir_examples/braille_noDelay_bias_zero.nir")