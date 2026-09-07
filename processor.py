import abc
import json
import logging
from typing import Dict, List

import datasets as ds
from PIL import Image
from tqdm.auto import tqdm

from .annotation import AnnotationData
from .category import CategoryData
from .image import ImageData
from .license import LicenseData
from .typehint import CategoryId, ImageId, JsonDict, LicenseId, PilImage

logger = logging.getLogger(__name__)


class MsCocoProcessor(object, metaclass=abc.ABCMeta):
    def load_image(self, image_path: str) -> PilImage:
        return Image.open(image_path)

    def load_annotation_json(self, ann_file_path: str) -> JsonDict:
        logger.info(f"Load annotation json from {ann_file_path}")
        with open(ann_file_path, "r") as rf:
            ann_json = json.load(rf)
        return ann_json

    def load_licenses_data(
        self, license_dicts: List[JsonDict]
    ) -> Dict[LicenseId, LicenseData]:
        licenses = {}
        for license_dict in license_dicts:
            license_data = LicenseData.from_dict(license_dict)
            licenses[license_data.license_id] = license_data
        return licenses

    def load_images_data(
        self,
        image_dicts: List[JsonDict],
        tqdm_desc: str = "Load images",
    ) -> Dict[ImageId, ImageData]:
        images = {}
        for image_dict in tqdm(image_dicts, desc=tqdm_desc):
            image_data = ImageData.from_dict(image_dict)
            images[image_data.image_id] = image_data
        return images

    def load_categories_data(
        self,
        category_dicts: List[JsonDict],
        tqdm_desc: str = "Load categories",
    ) -> Dict[CategoryId, CategoryData]:
        categories = {}
        for category_dict in tqdm(category_dicts, desc=tqdm_desc):
            category_data = CategoryData.from_dict(category_dict)
            categories[category_data.category_id] = category_data
        return categories

    def get_features_base_dict(self):
        return {
            "image_id": ds.Value("int64"),
            "image": ds.Image(),
            "file_name": ds.Value("string"),
            "coco_url": ds.Value("string"),
            "height": ds.Value("int32"),
            "width": ds.Value("int32"),
            "date_captured": ds.Value("string"),
            "flickr_url": ds.Value("string"),
            "license_id": ds.Value("int32"),
            "license": {
                "url": ds.Value("string"),
                "license_id": ds.Value("int8"),
                "name": ds.Value("string"),
            },
        }

    @abc.abstractmethod
    def get_features(self, *args, **kwargs) -> ds.Features:
        raise NotImplementedError

    @abc.abstractmethod
    def load_data(self, ann_dicts: List[JsonDict], tqdm_desc: str = "", **kwargs):
        assert tqdm_desc != "", "tqdm_desc must be provided."
        raise NotImplementedError

    @abc.abstractmethod
    def generate_examples(
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[AnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        **kwargs,
    ):
        raise NotImplementedError
