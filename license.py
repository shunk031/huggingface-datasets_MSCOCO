from dataclasses import dataclass

from .typehint import JsonDict, LicenseId


@dataclass
class LicenseData(object):
    url: str
    license_id: LicenseId
    name: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "LicenseData":
        return cls(
            license_id=json_dict["id"],
            url=json_dict["url"],
            name=json_dict["name"],
        )
