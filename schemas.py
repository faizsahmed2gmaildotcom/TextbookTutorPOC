from pydantic import BaseModel as _BaseModel


class _Section(_BaseModel):
    name: str
    start: int


class _Chapter(_BaseModel):
    name: str
    start: int
    sections: dict[str, _Section]


class TOCStructure(_BaseModel):
    chapters: dict[int, _Chapter]


class _VisualElem(_BaseModel):
    name: str
    contents: str


class _Page(_BaseModel):
    visual_elems: list[_VisualElem]


class Pages(_BaseModel):
    pages: dict[int, _Page]
