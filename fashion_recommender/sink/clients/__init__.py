from dataclasses import dataclass
from typing import List

@dataclass
class FieldConfig:
    name: str
    primary_key: bool

@dataclass
class VectorField(FieldConfig):
    vec_size: int

@dataclass
class StringField(FieldConfig):
    max_length: int


@dataclass
class StructureConfig:
    fields: List[FieldConfig]
    name: str
    description: str
    index_by: str = None
    partitions: List[str] = None
