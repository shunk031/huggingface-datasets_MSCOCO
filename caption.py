import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Tuple, TypedDict

import datasets as ds
from tqdm.auto import tqdm

from .annotation import AnnotationData
from .base_example import BaseExample
from .image import ImageData
from .license import LicenseData
from .processor import MsCocoProcessor
from .typehint import AnnotationId, ImageId, JsonDict, LicenseId


@dataclass
class CaptionsAnnotationData(AnnotationData):
    caption: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CaptionsAnnotationData":
        return cls(
            annotation_id=json_dict["id"],
            image_id=json_dict["image_id"],
            caption=json_dict["caption"],
        )


class CaptionAnnotationDict(TypedDict):
    annotation_id: AnnotationId
    caption: str


class CaptionExample(BaseExample):
    annotations: List[CaptionAnnotationDict]


class CaptionsProcessor(MsCocoProcessor):
    def get_features(self, *args, **kwargs) -> ds.Features:
        features_dict = self.get_features_base_dict()
        annotations = ds.Sequence(
            {
                "annotation_id": ds.Value("int64"),
                "image_id": ds.Value("int64"),
                "caption": ds.Value("string"),
            }
        )
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_data(
        self,
        ann_dicts: List[JsonDict],
        tqdm_desc: str = "Load captions data",
        **kwargs,
    ) -> Dict[ImageId, List[CaptionsAnnotationData]]:
        annotations = defaultdict(list)
        for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
            ann_data = CaptionsAnnotationData.from_dict(ann_dict)
            annotations[ann_data.image_id].append(ann_data)
        return annotations

    def generate_examples(
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[CaptionsAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        **kwargs,
    ) -> Iterator[Tuple[int, CaptionExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            assert len(image_anns) > 0

            image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example = asdict(image_data)
            example["image"] = image
            example["license"] = asdict(licenses[image_data.license_id])

            example["annotations"] = []
            for ann in image_anns:
                example["annotations"].append(asdict(ann))

            yield idx, example  # type: ignore
