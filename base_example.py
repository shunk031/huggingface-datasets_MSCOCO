from typing import TypedDict

from .typehint import ImageId, LicenseId, PilImage


class LicenseDict(TypedDict):
    license_id: LicenseId
    name: str
    url: str


class BaseExample(TypedDict):
    image_id: ImageId
    image: PilImage
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
    license_id: LicenseId
    license: LicenseDict
