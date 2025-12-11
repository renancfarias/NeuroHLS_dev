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

    def create_header_file(self, file_name: str):

        file_content = "#pragma once\n\n"

        file_content += self._includes + "\n"
        file_content += self._code

        self._create_file(f"{file_name}.h", file_content)

    def create_code_file(self, file_name: str):
        
        file_content = self._includes + "\n"
        file_content += self._code

        self._create_file(f"{file_name}.cpp", file_content)

    def _create_file(self, file_name: str, file_content: str):
        
        Path(self._folder_path).mkdir(parents=True, exist_ok=True)

        with open(f"{self._folder_path}/{file_name}", "w") as f:
            f.write(file_content)
