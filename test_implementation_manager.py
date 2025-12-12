from NeuroHls.ImplementationManager import ImplementationManager
from NeuroHls.ModelConfig import *


def test_dense():

    model_config = ModelConfig()

    model_config.add_layer(DenseLayerConfig(784, 128))
    model_config.add_layer(DenseLayerConfig(128, 10))

    print(model_config)

    impl_manager = ImplementationManager("z_test")

    impl_manager.create_files_from_config(model_config)

    # test_cpp = ImplementationManager(784)

    # test_cpp._dense(784, 128, "potential_t")
    # test_cpp._dense(128, 10, "potential_t", is_output_layer=True)
    
    # test_cpp.generate_files("gen_test")

# test_conv()
test_dense()