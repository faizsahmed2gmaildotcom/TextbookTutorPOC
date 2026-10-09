from pydantic import BaseModel

chapter = {
    "name": "NULL",
    "start": -1,
    "end": -1,
    "sections": {}
}

section = {
    "name": "NULL",
    "start": -1,
    "end": -1
}


class _Section(BaseModel):
    name: str
    start: int
    end: int


class _Chapter(BaseModel):
    name: str
    start: int
    end: int
    sections: dict[int, _Section]


class TOCStructure(BaseModel):
    chapters: dict[int, _Chapter]
