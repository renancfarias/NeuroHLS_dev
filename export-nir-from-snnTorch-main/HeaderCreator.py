from pathlib import Path

class HeaderCreator:
    
    def __init__(self, file_name: str, folder_path: str):
        
        self._includes = ""
        self._code = ""
        self._folder_path = folder_path
        self._file_name = file_name

    def add_include(self, file_name: str):

        self._includes += f"#include \"{file_name}\"\n"

    def add_code(self, code: str):
        
        self._code += code

    def create_header(self):

        file_define_name = self._file_name.upper() + "_H_"

        final_str = f"#ifndef {file_define_name}\n#define {file_define_name}\n\n"

        final_str += self._includes + "\n"
        final_str += self._code

        final_str += "\n#endif"

        Path(self._folder_path).mkdir(parents=True, exist_ok=True)

        with open(f"{self._folder_path}/{self._file_name}.h", "w") as f:
            f.write(final_str)

def test_header_creator():

    header = HeaderCreator("test_header", "gen_test")

    header.add_include("include_aleatorio.h")

    header.add_code("int x = 10;")

    header.create_header()

test_header_creator()

