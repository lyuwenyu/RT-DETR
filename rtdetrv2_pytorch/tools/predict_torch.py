import numpy as np
import argparse
from pathlib import Path
import sys
import time
from tqdm import tqdm
import cv2
import json
import pickle

sys.path.append('/home/ros/RT-DETR')
from src.core import YAMLConfig 

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

class ImageReader:
    def __init__(self, resize=1280, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        self.resize = resize
        self.pil_img = None   

    def __call__(self, image_path, *args, **kwargs):
        self.pil_img = Image.open(image_path).convert('RGB')
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)

def inference(image_paths, id2cat):
    device = torch.device(args.device)
    reader = ImageReader()
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)
    torch.manual_seed(21)

    inf_times = []
    outputs = []
    for img_path in tqdm(image_paths, total=len(image_paths)):
        # print(f'inference for: {img_path}')

        img = reader(img_path).to(device)
        w,h = reader.pil_img.size
        size = torch.tensor([[w,h]]).to(device)
        
        with torch.no_grad():
            start_time = time.time()
            output = model(img, size)
            labels, boxes, scores = output # (Batch, Preds)

            # Batch size = 1
            scr = scores[0].cpu().numpy()
            if len(labels.shape) > 2:
                lab = labels[0].cpu().numpy()
            else:
                lab = np.array([id2cat[l] for l in labels[0].cpu().numpy()])
            box = boxes[0].cpu().numpy()

            outputs.append({"scores": scr, "labels": lab, "boxes": box})
            inf_times.append(time.time() - start_time)
    
    fps = 1/np.mean(inf_times)
    print(f"Inference time = {np.mean(inf_times):0.3f} s")
    print(f"FPS = {fps:0.2f} ")

    return outputs, inf_times

def draw_boxes(image_paths, output, score_th = 0.1):
    
    for img_path, output in zip(image_paths,outputs):
        im = cv2.imread(str(img_path))        

        for s, l, b in zip(output["scores"], output["labels"], output["boxes"]):
            if s < score_th:
                continue
            b = list(map(int, b))  # Convert box coordinates to integers
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=2)

            # Scale text size and thickness
            font_scale = max(0.4, min(im.shape[1], im.shape[0]) / 1000)  # Dynamic scaling
            font_thickness = max(1, int(font_scale))

            # Add label and score text
            if isinstance(l, np.ndarray):
                # label_text = str([np.round(p,2) for p in l.tolist()])
                label_text = str(np.round(s,2))
            else:
                label_text = f"{id2cat[l]}: {s:.2f}"
            text_size = cv2.getTextSize(label_text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness)[0]
            text_origin = (b[0], b[1] - 10 if b[1] - 10 > 10 else b[1] + 10 + text_size[1])

            cv2.putText(im, label_text, text_origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)

        save_path = Path("vis_results") / img_path.name
        save_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_path), im)
 
def read_coco(coco_json_path: str, data_dir: str):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    image_paths = []
    for image_entry in coco_data.get('images', []):
        file_name = image_entry.get('file_name')
        if file_name:
            image_path = data_dir / Path(file_name)

            assert image_path.exists(), (f"Image does not exist: {image_path}")
            image_paths.append(str(image_path))
    
    id2cat = {cat['id']:cat['name'] for cat in coco_data.get('categories')}
    return image_paths, id2cat

def save_preds(image_paths, outputs, data_dir, save_path):
    results = {}
    for img_path, output in zip(image_paths,outputs):
        rel_img_path = Path(img_path).resolve().relative_to(data_dir)
        results[str(rel_img_path)] = output

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str)
    parser.add_argument("--ckpt", '-w', type=str) # pth
    parser.add_argument("--images", '-i', type=str, help='folder of images', default=None)
    parser.add_argument("--coco", type=str, help='inference on the image files from a coco json', default=None)
    parser.add_argument("--device", default="cuda:0")

    # TODO: Implement this
    parser.add_argument("--batch-size", type=int, default=1)

    # Save preds
    parser.add_argument("--data-dir", default="/home/ros/RT-DETR", help="Path to data folder")
    parser.add_argument("--save_path", default="/home/ros/RT-DETR/output2.pkl", help="Path to output file")

    args = parser.parse_args()

    if args.coco is not None:
        image_paths, id2cat = read_coco(args.coco, args.data_dir)
    elif args.images is not None:
        image_paths = [p for p in Path(args.images).glob('*') if p.suffix in [".jpg", ".png"]]
        id2cat = {
            1:"marine_mammal",
            2:"marker",
            3:"unknown",
            4:"vessel",
        }
    else:
        raise ValueError("Specify either coco json path or image directory")

    outputs, inf_times = inference(image_paths, id2cat)

    # save_preds(image_paths, outputs, args.data_dir, args.save_path)

    # draw cv2 boxes
    draw_boxes(image_paths, outputs)



    