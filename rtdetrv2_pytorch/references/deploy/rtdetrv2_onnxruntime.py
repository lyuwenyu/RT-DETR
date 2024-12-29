"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision.transforms as T

import numpy as np 
import onnxruntime as ort 
from PIL import Image, ImageDraw


def draw(images, labels, boxes, scores, thrh = 0.7):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for b,l,s in zip(box, lab, scr):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(np.round(l,2)), fill='blue', )
            draw.text((b[2], b[3]), text=str(np.round(s,2)), fill='black', )

        # for b, l in zip(box,scr):
        #     draw.rectangle(list(b), outline='red',)
        #     draw.text((b[0], b[1]), text=f"{l.item():0.2f}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    sess = ort.InferenceSession(args.onnx_file)
    print(ort.get_device())

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose([
        T.Resize((1280, 1280)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None]

    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
    )

    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-file', type=str, )
    parser.add_argument('--im-file', type=str, )
    # parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
