from typing import Annotated, Any, Dict, List, Literal, Tuple, TypedDict

from PIL.Image import Image

JsonDict = Dict[str, Any]
ImageId = int
AnnotationId = int
LicenseId = int
CategoryId = int
Bbox = Tuple[float, float, float, float]

MscocoSplits = Literal["train", "val", "test"]

PilImage = Annotated[Image, "Pillow Image"]


class UncompressedRLE(TypedDict):
    counts: List[int]
    size: Tuple[int, int]


class CompressedRLE(TypedDict):
    counts: bytes
    size: Tuple[int, int]


class CategoryDict(TypedDict):
    category_id: CategoryId
    name: str
    supercategory: str
