from pathlib import Path

class CodeCreator:
    
    def __init__(self, file_name: str, folder_path: str):
        
        self._includes = ""
        self._code = ""
        self._folder_path = folder_path
        self._file_name = file_name

    def add_include(self, file_name: str):

        self._includes += f"#include \"{file_name}\"\n"

    def add_code(self, code: str):
        
        self._code += code

    # GET_CPP -> RETORNA CODIGO
    # GET_H -> RETORNA CÓDIGO + DEFINES _H_

    def get_cpp(self):
        
        return self._includes + self._code

    def get_h(self):
        pass

    def create_header(self):

        # REMOVER METODO

        file_define_name = self._file_name.upper() + "_H_"

        final_str = f"#ifndef {file_define_name}\n#define {file_define_name}\n\n"

        final_str += self._includes + "\n"
        final_str += self._code

        final_str += "\n#endif"

        Path(self._folder_path).mkdir(parents=True, exist_ok=True)

        with open(f"{self._folder_path}/{self._file_name}.h", "w") as f:
            f.write(final_str)

# def test_header_creator():

#     header = HeaderCreator("test_header", "gen_test")

#     header.add_include("include_aleatorio.h")

#     header.add_code("int x = 10;")

#     header.create_header()

# test_header_creator()

