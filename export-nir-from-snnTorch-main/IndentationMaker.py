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