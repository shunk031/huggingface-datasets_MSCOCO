import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Final, Iterator, List, Tuple, TypedDict

import datasets as ds
from tqdm.auto import tqdm

from .base_example import BaseExample
from .category import CategoryData
from .image import ImageData
from .instances import (
    InstanceAnnotationDict,
    InstancesAnnotationData,
    InstancesProcessor,
)
from .license import LicenseData
from .typehint import CategoryId, ImageId, JsonDict, LicenseId

KEYPOINT_STATE: Final[List[str]] = ["unknown", "invisible", "visible"]


@dataclass
class PersonKeypoint(object):
    x: int
    y: int
    v: int
    state: str


@dataclass
class PersonKeypointsAnnotationData(InstancesAnnotationData):
    num_keypoints: int
    keypoints: List[PersonKeypoint]

    @classmethod
    def v_keypoint_to_state(cls, keypoint_v: int) -> str:
        return KEYPOINT_STATE[keypoint_v]

    @classmethod
    def get_person_keypoints(
        cls, flatten_keypoints: List[int], num_keypoints: int
    ) -> List[PersonKeypoint]:
        keypoints_x = flatten_keypoints[0::3]
        keypoints_y = flatten_keypoints[1::3]
        keypoints_v = flatten_keypoints[2::3]
        assert len(keypoints_x) == len(keypoints_y) == len(keypoints_v)

        keypoints = [
            PersonKeypoint(x=x, y=y, v=v, state=cls.v_keypoint_to_state(v))
            for x, y, v in zip(keypoints_x, keypoints_y, keypoints_v)
        ]
        assert len([kp for kp in keypoints if kp.state != "unknown"]) == num_keypoints
        return keypoints

    @classmethod
    def from_dict(
        cls,
        json_dict: JsonDict,
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
    ) -> "PersonKeypointsAnnotationData":
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
        flatten_keypoints = json_dict["keypoints"]
        num_keypoints = json_dict["num_keypoints"]
        keypoints = cls.get_person_keypoints(flatten_keypoints, num_keypoints)

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
            #
            # PersonKeypointsAnnotationData
            #
            num_keypoints=num_keypoints,
            keypoints=keypoints,
        )


class KeypointDict(TypedDict):
    x: int
    y: int
    v: int
    state: str


class PersonKeypointAnnotationDict(InstanceAnnotationDict):
    num_keypoints: int
    keypoints: List[KeypointDict]


class PersonKeypointExample(BaseExample):
    annotations: List[PersonKeypointAnnotationDict]


class PersonKeypointsProcessor(InstancesProcessor):
    def get_features(self, decode_rle: bool) -> ds.Features:
        features_dict = self.get_features_base_dict()
        features_instance_dict = self.get_features_instance_dict(decode_rle=decode_rle)
        features_instance_dict.update(
            {
                "keypoints": ds.Sequence(
                    {
                        "state": ds.Value("string"),
                        "x": ds.Value("int32"),
                        "y": ds.Value("int32"),
                        "v": ds.Value("int32"),
                    }
                ),
                "num_keypoints": ds.Value("int32"),
            }
        )
        annotations = ds.Sequence(features_instance_dict)
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_data(  # type: ignore[override]
        self,
        ann_dicts: List[JsonDict],
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
        tqdm_desc: str = "Load person keypoints data",
    ) -> Dict[ImageId, List[PersonKeypointsAnnotationData]]:
        annotations = defaultdict(list)
        ann_dicts = sorted(ann_dicts, key=lambda d: d["image_id"])

        for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
            ann_data = PersonKeypointsAnnotationData.from_dict(
                ann_dict, images=images, decode_rle=decode_rle
            )
            annotations[ann_data.image_id].append(ann_data)
        return annotations

    def generate_examples(  # type: ignore[override]
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[PersonKeypointsAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        categories: Dict[CategoryId, CategoryData],
    ) -> Iterator[Tuple[int, PersonKeypointExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            if len(image_anns) < 1:
                # If there are no persons in the image,
                # no keypoint annotations will be assigned.
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
