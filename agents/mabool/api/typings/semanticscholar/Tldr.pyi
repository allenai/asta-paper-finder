from semanticscholar.SemanticScholarObject import (
    SemanticScholarObject as SemanticScholarObject,
)

class Tldr(SemanticScholarObject):
    def __init__(self, data) -> None: ...
    @property
    def model(self) -> str: ...
    @property
    def text(self) -> str: ...
