import re

from ..parser.ast_transformer import build_ast
from ..parser.cfg_builder import get_cfg
from ..parser.def_use import AssignsVisitor, DefUseChains
from ..parser.lark_parser import parse_to_tree


class IntermediateRepresentations:
    def __init__(self, cell_code = "", cell_id = None, last_ran_code = "") -> None:
        self.cell_id = cell_id
        self.cell_code = self.remove_magic(cell_code)
        self.last_ran_code = last_ran_code
        self.update_AST()
        self.update_cfg()
        self.update_assigns()

    def remove_magic(self, cell_code):
        new_cell_code = ""
        pat1 = "^[!\\%]"
        for ds in cell_code.splitlines():
            if (not re.match(pat1, ds)):
                if (new_cell_code == ""):
                    new_cell_code = ds
                else:
                    new_cell_code = new_cell_code + "\n" + ds
        return new_cell_code

    def update_AST(self):
        self.AST = build_ast(parse_to_tree(self.cell_code))

    def update_cfg(self):
        self.CFG = get_cfg(self.AST, self.cell_id)

    def update_assigns(self):
        defUseChains = DefUseChains()
        defUseChains.visit(self.AST, self.CFG)
        self.UDA = AssignsVisitor(defUseChains)
        self.UDA.visit(self.AST)
        self.UDA.combine()

    def __eq__(self, __o: object) -> bool:
        return self.cell_id == __o.cell_id and self.cell_code == __o.cell_code

    def __str__(self) -> str:
        return f"index: {self.cell_id}, code: {self.cell_code}"
