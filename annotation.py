from dataclasses import dataclass

from .typehint import AnnotationId, ImageId


@dataclass
class AnnotationData(object):
    annotation_id: AnnotationId
    image_id: ImageId
