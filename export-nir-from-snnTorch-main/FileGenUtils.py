from pathlib import Path
import shutil
import os

class IndentationMaker:
    
    def __init__(self, tab_count : int, first_line_should_use_indentation = True):
        self._str = ""
        self._tab_count = tab_count
        self.first_line_should_use_indentation = first_line_should_use_indentation
        self._scope_count = 0

    def append_line(self, line, should_break_line = True):

        if self.first_line_should_use_indentation or self._str != "":
            for i in range(self._tab_count):
                self._str += "\t"

        self._str += line

        if should_break_line:
            self._str += "\n"

    def add_scope(self):
        self.append_line("{")
        self._scope_count += 1
        self._tab_count += 1

    def _close_braces(self):

        if self._scope_count == 0:
            return
        
        for i in range(self._scope_count - 1):
            self._tab_count -= 1
            self.append_line("}")

        self._tab_count -= 1
        self.append_line("}", should_break_line=False)

    def get_text(self):
        self._close_braces()
        return self._str


def get_bracket_notation_of_tuple(shape : tuple):

    if not isinstance(shape, tuple):
        shape = (shape, )

    brackets = ""

    for i in shape:
        brackets += f"[{i}]"
    
    return brackets

def copy_folder_from_backend(path_inside_backend, to_path):

    backend_path = f"backend/{path_inside_backend}"

    if not os.path.exists(backend_path):
        raise Exception(f"Folder {backend_path} does not exist")

    to_path = f"{to_path}/{path_inside_backend}"
    Path(to_path).mkdir(parents=True, exist_ok=True)

    shutil.copytree(backend_path, to_path, dirs_exist_ok=True)

def copy_file_from_backend(path_inside_backend, to_path):
    
    backend_path = f"backend/{path_inside_backend}"

    if not os.path.exists(backend_path):
        raise Exception(f"File {backend_path} does not exist")

    Path(to_path).mkdir(parents=True, exist_ok=True)

    shutil.copy(backend_path, to_path)
    