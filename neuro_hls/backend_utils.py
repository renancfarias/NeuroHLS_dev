from pathlib import Path
import shutil
import os

def get_backend_folder():
    base_dir = os.path.dirname(__file__)
    backend_folder = os.path.join(base_dir, "backend")

    return backend_folder

def copy_backend_to(folder_path: str):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    backend_folder = get_backend_folder()

    for item in os.listdir(backend_folder):

        if item == "testbench.cpp":
            continue
        
        source_path = os.path.join(backend_folder, item)
        target_path = os.path.join(folder_path, item)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)

def get_testbench_content():

    backend_folder = get_backend_folder()   
    testbench_path = os.path.join(backend_folder, "testbench.cpp")

    with open(testbench_path, "r", encoding="utf-8") as f:
        tb_cpp = f.read()

    return tb_cpp
    