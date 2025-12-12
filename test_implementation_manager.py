from NeuroHls.ImplementationManager import ImplementationManager
from NeuroHls.ModelConfig import *


def test_dense():

    model_config = ModelConfig()

    model_config.add_layer(DenseLayerConfig(784, 128))
    model_config.add_layer(DenseLayerConfig(128, 10))

    model_config.set_input_quantization(10, 2)
    model_config.set_default_potential_quantization(25, 6)

    model_config.layers[0].set_potential_quantization(43, 10)

    print(model_config)

    impl_manager = ImplementationManager("z_test")

    impl_manager.create_files_from_config(model_config)

test_dense()