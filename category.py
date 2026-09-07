from dataclasses import dataclass

from .typehint import JsonDict


@dataclass
class CategoryData(object):
    category_id: int
    name: str
    supercategory: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CategoryData":
        return cls(
            category_id=json_dict["id"],
            name=json_dict["name"],
            supercategory=json_dict["supercategory"],
        )
