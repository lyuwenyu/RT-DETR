import numpy as np
from PIL import Image
import supervisely as sly
from torchvision import datapoints
import torchvision.transforms.v2 as T
from src.core import register


@register
class ImgAug(T.Transform):
    def __init__(self, config_path):
        self.config_path = config_path
        config = sly.json.load_json_file(self.config_path)
        self.augs = sly.imgaug_utils.build_pipeline(
            config["pipeline"], random_order=config["random_order"]
        )

        sly.logger.debug(
            "ImgAug loaded: ",
            extra=dict(config_path=self.config_path, pipeline=config["pipeline"]),
        )

    def __call__(self, inputs):
        img = inputs[0]
        img = np.asarray(img)
        target = inputs[1]
        boxes : datapoints.BoundingBox = inputs[1]['boxes']
        boxes = boxes.tolist()
        img, boxes = sly.imgaug_utils.apply_to_image_and_bbox(self.augs, img, boxes)
        target['boxes'] = datapoints.BoundingBox(
            np.array(boxes, np.float32),
            format=datapoints.BoundingBoxFormat.XYXY, 
            spatial_size=img.shape[:2]) # h w
        img = Image.fromarray(img)
        return img, target

    def __repr__(self):
        return str(self.augs)