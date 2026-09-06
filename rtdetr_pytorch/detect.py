import torch
import cv2
from PIL import Image
from torchvision import transforms

# load model
model = torch.load("weights/rtdetr_r50vd_6x_coco_from_paddle.pth")
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# load image
img = Image.open("test.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# run inference
with torch.no_grad():
    outputs = model(input_tensor)

print(outputs)