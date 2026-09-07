from dataclasses import dataclass

from .typehint import JsonDict


@dataclass
class AnnotationInfo(object):
    description: str
    url: str
    version: str
    year: str
    contributor: str
    date_created: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "AnnotationInfo":
        return cls(**json_dict)
