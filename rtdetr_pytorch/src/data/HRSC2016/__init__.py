import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from src.core import register
from src.data.coco.coco_rotated_dataset import RotatedCocoDataset, BoundingBox

print("Registering HRSC2016Dataset")
@register
class HRSC2016Dataset(RotatedCocoDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_folder,
        transforms,
        return_masks,
        remap_mscoco_category=False,
    ):
        img_dir = Path(img_folder).resolve()
        ann_dir = Path(ann_folder).resolve()
        assert img_dir.is_dir(), f"No directory found at {img_dir}"
        assert ann_dir.is_dir(), f"No directory found at {ann_dir}"
        image_files: List[Path] = list(Path(img_folder).iterdir())
        labels: List[List[BoundingBox]] = []
        for file in Path(ann_folder).iterdir():
            labels.append(self.parse_bboxes(file))

        super(HRSC2016Dataset, self).__init__(
            image_files=image_files,
            labels=labels,
        )

    def parse_bboxes(self, file: Path):
        hrsc_dict = self.parse_hrsc2016_xml(file.read_text())
        bboxes = []
        for object in hrsc_dict['objects']:
            bboxes.append(
                BoundingBox(
                    category="object",
                    x1=object['robndbox']['cx'] - object['robndbox']['w'] / 2,
                    x2=object['robndbox']['cx'] + object['robndbox']['w'] / 2,
                    y1=object['robndbox']['cy'] - object['robndbox']['h'] / 2,
                    y2=object['robndbox']['cy'] + object['robndbox']['h'] / 2,
                    rotation=object['robndbox']['angle'] * 180 / math.pi
                )
            )
        return bboxes



    def parse_hrsc2016_xml(self, xml_string):
        root = ET.fromstring(xml_string)

        annotation_dict = {}
        annotation_dict['verified'] = root.attrib.get('verified', 'no')
        annotation_dict['folder'] = root.find('folder').text
        annotation_dict['filename'] = root.find('filename').text
        annotation_dict['path'] = root.find('path').text

        source = root.find('source')
        annotation_dict['source'] = {
            'database': source.find('database').text
        }

        size = root.find('size')
        annotation_dict['size'] = {
            'width': int(size.find('width').text),
            'height': int(size.find('height').text),
            'depth': int(size.find('depth').text)
        }

        annotation_dict['segmented'] = int(root.find('segmented').text)

        objects = []
        for obj in root.findall('object'):
            object_dict = {
                'type': obj.find('type').text,
                'name': obj.find('name').text,
                'pose': obj.find('pose').text,
                'truncated': int(obj.find('truncated').text),
                'difficult': int(obj.find('difficult').text),
                'bndbox': {
                    'xmin': int(obj.find('bndbox/xmin').text),
                    'ymin': int(obj.find('bndbox/ymin').text),
                    'xmax': int(obj.find('bndbox/xmax').text),
                    'ymax': int(obj.find('bndbox/ymax').text)
                },
                'robndbox': {
                    'cx': float(obj.find('robndbox/cx').text),
                    'cy': float(obj.find('robndbox/cy').text),
                    'w': float(obj.find('robndbox/w').text),
                    'h': float(obj.find('robndbox/h').text),
                    'angle': float(obj.find('robndbox/angle').text)
                }
            }
            objects.append(object_dict)

        annotation_dict['objects'] = objects

        return annotation_dict