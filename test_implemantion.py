from NeuroHls.ImplementationManager import ImplementationManager


def test_dense():

    test_cpp = ImplementationManager(784)

    test_cpp.dense(784, 128, "potential_t")
    test_cpp.dense(128, 10, "potential_t", is_output_layer=True)
    
    test_cpp.generate_files("gen_test")

# test_conv()
test_dense()