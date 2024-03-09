import supervisely as sly
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def convert_dataset_to_coco(dataset: sly.Dataset, meta: sly.ProjectMeta, selected_classes: list) -> COCO:
    coco_anno = {"images": [], "categories": [], "annotations": []}
    cat2id = {name: i + 1 for i, name in enumerate(selected_classes)}
    img_id = 1
    ann_id = 1
    for name in dataset.get_items_names():
        ann = dataset.get_ann(name, meta)
        img_dict = {
            "id": img_id,
            "height": ann.img_size[0],
            "width": ann.img_size[1],
            "file_name": name
        }
        coco_anno["images"].append(img_dict)
        
        for label in ann.labels:
            if isinstance(label.geometry, (sly.Bitmap, sly.Polygon)):
                rect = label.geometry.to_bbox()
            elif isinstance(label.geometry, sly.Rectangle):
                rect = label.geometry
            else:
                pass
            x,y,x2,y2 = rect.left, rect.top, rect.right, rect.bottom
            ann_dict = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat2id[label.obj_class.name],
                "bbox": [x, y, x2 - x, y2 - y],
                "area": (x2 - x) * (y2 - y),
                "iscrowd": 0
            }
            coco_anno["annotations"].append(ann_dict)
            ann_id += 1
        
        img_id += 1

    coco_anno["categories"] = [{"id": i, "name": name} for name, i in cat2id.items()]
    coco_api = COCO()
    coco_api.dataset = coco_anno
    coco_api.createIndex()
    return coco_api
