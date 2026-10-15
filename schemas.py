from pydantic import BaseModel as _BaseModel


class _Section(_BaseModel):
    name: str
    start: int


class _Chapter(_BaseModel):
    name: str
    start: int
    sections: dict[str, _Section]


class TOCStructure(_BaseModel):
    chapters: dict[str, _Chapter]
    thinking: str


class _VisualElem(_BaseModel):
    name: str
    contents: str


class _Page(_BaseModel):
    visual_elems: list[_VisualElem]


class Pages(_BaseModel):
    pages: dict[int, _Page]


class Intent(_BaseModel):
    sufficient: bool
    implicit_reference: bool
    implicit_target: str
    additional_targets: list[str]
    reasoning: str
