import logging
import os
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import datasets as ds
from datasets.data_files import DataFilesDict

from .caption import CaptionsProcessor
from .instances import InstancesProcessor
from .person_keypoint import PersonKeypointsProcessor
from .processor import MsCocoProcessor
from .typehint import MscocoSplits

logger = logging.getLogger(__name__)


_CITATION = """\
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}
"""

_DESCRIPTION = """\
COCO is a large-scale object detection, segmentation, and captioning dataset.
"""

_HOMEPAGE = """\
https://cocodataset.org/#home
"""

_LICENSE = """\
The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License.
"""

_URLS = {
    "2014": {
        "images": {
            "train": "http://images.cocodataset.org/zips/train2014.zip",
            "validation": "http://images.cocodataset.org/zips/val2014.zip",
            "test": "http://images.cocodataset.org/zips/test2014.zip",
        },
        "annotations": {
            "train_validation": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "test_image_info": "http://images.cocodataset.org/annotations/image_info_test2014.zip",
        },
    },
    "2015": {
        "images": {
            "test": "http://images.cocodataset.org/zips/test2015.zip",
        },
        "annotations": {
            "test_image_info": "http://images.cocodataset.org/annotations/image_info_test2015.zip",
        },
    },
    "2017": {
        "images": {
            "train": "http://images.cocodataset.org/zips/train2017.zip",
            "validation": "http://images.cocodataset.org/zips/val2017.zip",
            "test": "http://images.cocodataset.org/zips/test2017.zip",
            "unlabeled": "http://images.cocodataset.org/zips/unlabeled2017.zip",
        },
        "annotations": {
            "train_validation": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "stuff_train_validation": "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
            "panoptic_train_validation": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
            "test_image_info": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
            "unlabeled": "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
        },
    },
}


class MsCocoConfig(ds.BuilderConfig):
    YEARS: Tuple[int, ...] = (
        2014,
        2017,
    )
    TASKS: Tuple[str, ...] = (
        "captions",
        "instances",
        "person_keypoints",
    )

    def __init__(
        self,
        year: int,
        coco_task: Union[str, Sequence[str]],
        version: Optional[Union[ds.Version, str]],
        decode_rle: bool = False,
        data_dir: Optional[str] = None,
        data_files: Optional[DataFilesDict] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=self.config_name(year=year, task=coco_task),
            version=version,
            data_dir=data_dir,
            data_files=data_files,
            description=description,
        )
        self._check_year(year)
        self._check_task(coco_task)

        self._year = year
        self._task = coco_task
        self.processor = self.get_processor()
        self.decode_rle = decode_rle

    def _check_year(self, year: int) -> None:
        assert year in self.YEARS, year

    def _check_task(self, task: Union[str, Sequence[str]]) -> None:
        if isinstance(task, str):
            assert task in self.TASKS, task
        elif isinstance(task, list) or isinstance(task, tuple):
            for t in task:
                assert t, task
        else:
            raise ValueError(f"Invalid task: {task}")

    @property
    def year(self) -> int:
        return self._year

    @property
    def task(self) -> str:
        if isinstance(self._task, str):
            return self._task
        elif isinstance(self._task, list) or isinstance(self._task, tuple):
            return "-".join(sorted(self._task))
        else:
            raise ValueError(f"Invalid task: {self._task}")

    def get_processor(self) -> MsCocoProcessor:
        if self.task == "captions":
            return CaptionsProcessor()
        elif self.task == "instances":
            return InstancesProcessor()
        elif self.task == "person_keypoints":
            return PersonKeypointsProcessor()
        else:
            raise ValueError(f"Invalid task: {self.task}")

    @classmethod
    def config_name(cls, year: int, task: Union[str, Sequence[str]]) -> str:
        if isinstance(task, str):
            return f"{year}-{task}"
        elif isinstance(task, list) or isinstance(task, tuple):
            task = "-".join(task)
            return f"{year}-{task}"
        else:
            raise ValueError(f"Invalid task: {task}")


def dataset_configs(year: int, version: ds.Version) -> List[MsCocoConfig]:
    return [
        MsCocoConfig(
            year=year,
            coco_task="captions",
            version=version,
        ),
        MsCocoConfig(
            year=year,
            coco_task="instances",
            version=version,
        ),
        MsCocoConfig(
            year=year,
            coco_task="person_keypoints",
            version=version,
        ),
        # MsCocoConfig(
        #     year=year,
        #     coco_task=("captions", "instances"),
        #     version=version,
        # ),
        # MsCocoConfig(
        #     year=year,
        #     coco_task=("captions", "person_keypoints"),
        #     version=version,
        # ),
    ]


def configs_2014(version: ds.Version) -> List[MsCocoConfig]:
    return dataset_configs(year=2014, version=version)


def configs_2017(version: ds.Version) -> List[MsCocoConfig]:
    return dataset_configs(year=2017, version=version)


class MsCocoDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = MsCocoConfig
    BUILDER_CONFIGS = configs_2014(version=VERSION) + configs_2017(version=VERSION)

    @property
    def year(self) -> int:
        config: MsCocoConfig = self.config  # type: ignore
        return config.year

    @property
    def task(self) -> str:
        config: MsCocoConfig = self.config  # type: ignore
        return config.task

    def _info(self) -> ds.DatasetInfo:
        processor: MsCocoProcessor = self.config.processor  # type: ignore
        features = processor.get_features(decode_rle=self.config.decode_rle)  # type: ignore
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS[f"{self.year}"])

        imgs = file_paths["images"]  # type: ignore
        anns = file_paths["annotations"]  # type: ignore

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "base_image_dir": imgs["train"],
                    "base_annotation_dir": anns["train_validation"],
                    "split": "train",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "base_image_dir": imgs["validation"],
                    "base_annotation_dir": anns["train_validation"],
                    "split": "val",
                },
            ),
            # ds.SplitGenerator(
            #     name=ds.Split.TEST,  # type: ignore
            #     gen_kwargs={
            #         "base_image_dir": imgs["test"],
            #         "test_image_info_path": anns["test_image_info"],
            #         "split": "test",
            #     },
            # ),
        ]

    def _generate_train_val_examples(
        self, split: str, base_image_dir: str, base_annotation_dir: str
    ):
        image_dir = os.path.join(base_image_dir, f"{split}{self.year}")

        ann_dir = os.path.join(base_annotation_dir, "annotations")
        ann_file_path = os.path.join(ann_dir, f"{self.task}_{split}{self.year}.json")

        processor: MsCocoProcessor = self.config.processor  # type: ignore

        ann_json = processor.load_annotation_json(ann_file_path=ann_file_path)

        # info = AnnotationInfo.from_dict(ann_json["info"])
        licenses = processor.load_licenses_data(license_dicts=ann_json["licenses"])
        images = processor.load_images_data(image_dicts=ann_json["images"])

        category_dicts = ann_json.get("categories")
        categories = (
            processor.load_categories_data(category_dicts=category_dicts)
            if category_dicts is not None
            else None
        )

        config: MsCocoConfig = self.config  # type: ignore
        yield from processor.generate_examples(
            annotations=processor.load_data(
                ann_dicts=ann_json["annotations"],
                images=images,
                decode_rle=config.decode_rle,
            ),
            categories=categories,
            image_dir=image_dir,
            images=images,
            licenses=licenses,
        )

    def _generate_test_examples(self, test_image_info_path: str):
        raise NotImplementedError

    def _generate_examples(
        self,
        split: MscocoSplits,
        base_image_dir: Optional[str] = None,
        base_annotation_dir: Optional[str] = None,
        test_image_info_path: Optional[str] = None,
    ):
        if split == "test" and test_image_info_path is not None:
            yield from self._generate_test_examples(
                test_image_info_path=test_image_info_path
            )
        elif (
            split in get_args(MscocoSplits)
            and base_image_dir is not None
            and base_annotation_dir is not None
        ):
            yield from self._generate_train_val_examples(
                split=split,
                base_image_dir=base_image_dir,
                base_annotation_dir=base_annotation_dir,
            )
        else:
            raise ValueError(
                f"Invalid arguments: split = {split}, "
                f"base_image_dir = {base_image_dir}, "
                f"base_annotation_dir = {base_annotation_dir}, "
                f"test_image_info_path = {test_image_info_path}",
            )
