from read_nir import read_nir
from implementation_manager import *

from params_extraction_test import extract

def teste(nir_file):
    model = read_nir(nir_file)
    print(model)

    # print("IMPLEMENTATION:\n\n")

    implement_model(model)

    layer = model.layers[0]

    for l in model.layers:
        if l.name == "lif1.lif":
            layer = l
            continue
    
    print("\n\n\n")
    print(f"{layer.w_in}\n\n")
    print(f"r{get_bracket_str(layer.r.shape)} = {extract(layer.r)}")


# teste("z_nir_examples/lif_norse.nir")
# teste("z_nir_examples/cnn_sinabs.nir")
teste("z_nir_examples/braille_noDelay_bias_zero.nir")