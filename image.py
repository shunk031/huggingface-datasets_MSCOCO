from dataclasses import dataclass

from .typehint import ImageId, JsonDict, LicenseId


@dataclass
class ImageData(object):
    image_id: ImageId
    license_id: LicenseId
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ImageData":
        return cls(
            image_id=json_dict["id"],
            license_id=json_dict["license"],
            file_name=json_dict["file_name"],
            coco_url=json_dict["coco_url"],
            height=json_dict["height"],
            width=json_dict["width"],
            date_captured=json_dict["date_captured"],
            flickr_url=json_dict["flickr_url"],
        )
