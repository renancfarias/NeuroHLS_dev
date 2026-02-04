from pathlib import Path

class CodeCreator:
    
    def __init__(self, folder_path: str):
        
        self._includes = ""
        self._code = ""
        self._folder_path = folder_path

    def add_include(self, file_name: str):

        self._includes += f"#include \"{file_name}\"\n"

    def add_code(self, code: str):
        
        self._code += code

    def create_file(self, file_name: str):

        if file_name.split('.')[1] == 'h':
            file_content = "#pragma once\n\n"
        else:
            file_content = ""

        file_content += self._includes + "\n"
        file_content += self._code

        Path(self._folder_path).mkdir(parents=True, exist_ok=True)

        with open(f"{self._folder_path}/{file_name}", "w") as f:
            f.write(file_content)