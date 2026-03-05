import torch
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir

import subprocess
from pathlib import Path
import sys
import inspect

def get_caller_path():

    frame = inspect.stack()[2]
    filename = frame.filename

    return Path(filename).resolve().parent

def save_out_py(file_str):

    file = open(f"{get_caller_path()}/out_py.txt", "w")
    file.write(file_str)
    file.close()

def debug_net(net, sample_data, primitive_name):

    if not isinstance(sample_data, (list, tuple)):
        nir_sample_data = sample_data
        sample_data = [sample_data]
    else:
        nir_sample_data = sample_data[0]

    nir_graph = export_to_nir(net, nir_sample_data, model_name=primitive_name)
    
    net_from_nir = import_from_nir(nir_graph)
    net_from_nir.eval()

    print(f"\n*** {primitive_name} ***")

    print("\n" + "-" * 30 + "\nNIR Graph\n" + "-" * 30 + "\n")
    print(net_from_nir)

    # ----------------------------------------------------------

    print("\n" + "-" * 30 + "\nLayer Weights\n" + "-" * 30 + "\n")
    
    for name, layer in net_from_nir.named_children():

        if name == "input" or name == "output":
            continue

        print(f" - {layer}\n")

        for name, param in layer.named_parameters():
            print(f"\t{name}:")

            for tensor in param:
                print(f"\t\t{tensor.detach()}")

    # ----------------------------------------------------------

    print("\n" + "-" * 30 + "\nSamples\n" + "-" * 30 + "\n")

    print(f"Shape: {sample_data[0].shape[1:]}\n")

    for data in sample_data:
        print(data)

    # ----------------------------------------------------------

    print("\n" + "-" * 30 + "\nOutputs\n" + "-" * 30 + "\n")

    print(f"Shape: {net_from_nir(sample_data[0])[0].shape[1:]}\n")

    txt_str = ""

    for i, data in enumerate(sample_data):
        
        output = net_from_nir(data)
        print(output[0].detach())

        txt_str += f"\nSample {i + 1}:\n\n{output[0].detach().numpy()}\n"
    
    print()

    save_out_py(txt_str)

def run_cpp_impl():

    path = get_caller_path() / "impl.cpp"

    print('-' * 30 + '\n' + "C++ Implementation\n" + '-' * 30 + '\n')      
    print(f"Absolute Path: {path}\n")

    compile_cmd = ["g++", str(path), "-o", "exe_cpp"]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n\nErro na compilação:")
        print(result.stderr)
        return
    
    result = subprocess.run("./exe_cpp", capture_output=True, text=True)
    print(f"Output da impl em C++:\n\n{result.stdout}")

