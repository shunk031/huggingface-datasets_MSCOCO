import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Tuple, TypedDict, Union

import datasets as ds
import numpy as np
from pycocotools import mask as cocomask
from tqdm.auto import tqdm

from .annotation import AnnotationData
from .base_example import BaseExample
from .category import CategoryData
from .const import CATEGORIES, SUPER_CATEGORIES
from .image import ImageData
from .license import LicenseData
from .processor import MsCocoProcessor
from .typehint import (
    AnnotationId,
    Bbox,
    CategoryDict,
    CategoryId,
    CompressedRLE,
    ImageId,
    JsonDict,
    LicenseId,
    UncompressedRLE,
)

logger = logging.getLogger(__name__)


@dataclass
class InstancesAnnotationData(AnnotationData):
    segmentation: Union[np.ndarray, CompressedRLE]
    area: float
    iscrowd: bool
    bbox: Tuple[float, float, float, float]
    category_id: int

    @classmethod
    def compress_rle(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> CompressedRLE:
        if iscrowd:
            rle = cocomask.frPyObjects(segmentation, h=height, w=width)
        else:
            rles = cocomask.frPyObjects(segmentation, h=height, w=width)
            rle = cocomask.merge(rles)

        return rle  # type: ignore

    @classmethod
    def rle_segmentation_to_binary_mask(
        cls, segmentation, iscrowd: bool, height: int, width: int
    ) -> np.ndarray:
        rle = cls.compress_rle(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return cocomask.decode(rle)  # type: ignore

    @classmethod
    def rle_segmentation_to_mask(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> np.ndarray:
        binary_mask = cls.rle_segmentation_to_binary_mask(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return binary_mask * 255

    @classmethod
    def from_dict(
        cls,
        json_dict: JsonDict,
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
    ) -> "InstancesAnnotationData":
        segmentation = json_dict["segmentation"]
        image_id = json_dict["image_id"]
        image_data = images[image_id]
        iscrowd = bool(json_dict["iscrowd"])

        segmentation_mask = (
            cls.rle_segmentation_to_mask(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
            if decode_rle
            else cls.compress_rle(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
        )
        return cls(
            #
            # for AnnotationData
            #
            annotation_id=json_dict["id"],
            image_id=image_id,
            #
            # for InstancesAnnotationData
            #
            segmentation=segmentation_mask,  # type: ignore
            area=json_dict["area"],
            iscrowd=iscrowd,
            bbox=json_dict["bbox"],
            category_id=json_dict["category_id"],
        )


class InstanceAnnotationDict(TypedDict):
    annotation_id: AnnotationId
    area: float
    bbox: Bbox
    image_id: ImageId
    category_id: CategoryId
    category: CategoryDict
    iscrowd: bool
    segmentation: np.ndarray


class InstanceExample(BaseExample):
    annotations: List[InstanceAnnotationDict]


class InstancesProcessor(MsCocoProcessor):
    def get_features_instance_dict(self, decode_rle: bool):
        segmentation_feature = (
            ds.Image()
            if decode_rle
            else {
                "counts": ds.Sequence(ds.Value("int64")),
                "size": ds.Sequence(ds.Value("int32")),
            }
        )
        return {
            "annotation_id": ds.Value("int64"),
            "image_id": ds.Value("int64"),
            "segmentation": segmentation_feature,
            "area": ds.Value("float32"),
            "iscrowd": ds.Value("bool"),
            "bbox": ds.Sequence(ds.Value("float32"), length=4),
            "category_id": ds.Value("int32"),
            "category": {
                "category_id": ds.Value("int32"),
                "name": ds.ClassLabel(
                    num_classes=len(CATEGORIES),
                    names=CATEGORIES,
                ),
                "supercategory": ds.ClassLabel(
                    num_classes=len(SUPER_CATEGORIES),
                    names=SUPER_CATEGORIES,
                ),
            },
        }

    def get_features(self, decode_rle: bool) -> ds.Features:
        features_dict = self.get_features_base_dict()
        annotations = ds.Sequence(
            self.get_features_instance_dict(decode_rle=decode_rle)
        )
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_data(  # type: ignore[override]
        self,
        ann_dicts: List[JsonDict],
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
        tqdm_desc: str = "Load instances data",
    ) -> Dict[ImageId, List[InstancesAnnotationData]]:
        annotations = defaultdict(list)
        ann_dicts = sorted(ann_dicts, key=lambda d: d["image_id"])

        for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
            ann_data = InstancesAnnotationData.from_dict(
                ann_dict, images=images, decode_rle=decode_rle
            )
            annotations[ann_data.image_id].append(ann_data)

        return annotations

    def generate_examples(  # type: ignore[override]
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[InstancesAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        categories: Dict[CategoryId, CategoryData],
    ) -> Iterator[Tuple[int, InstanceExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            if len(image_anns) < 1:
                logger.warning(f"No annotation found for image id: {image_id}.")
                continue

            image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example = asdict(image_data)
            example["image"] = image
            example["license"] = asdict(licenses[image_data.license_id])

            example["annotations"] = []
            for ann in image_anns:
                ann_dict = asdict(ann)
                category = categories[ann.category_id]
                ann_dict["category"] = asdict(category)
                example["annotations"].append(ann_dict)

            yield idx, example  # type: ignore
