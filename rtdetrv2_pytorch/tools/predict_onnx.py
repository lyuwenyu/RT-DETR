import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import argparse
import time
from pathlib import Path

def read_img(img_path):
    im = Image.open(img_path).convert('RGB')
    im = im.resize((1280, 1280))
    im_data = ToTensor()(im)[None]
    # (width, height) = im.size
    # print(im_data.shape)
    # print(width, height)
    # size = torch.tensor([[width, height]])
    size = torch.tensor([[1280, 1280]])
    return im, im_data, size

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def main(args, ):
    
    print("ort.get_device()", ort.get_device())
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess = ort.InferenceSession(args.model, sess_options=sess_options, providers=providers)
    
    img_path_list = []
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png'] 
    for (root, dirs, files) in os.walk(args.img):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = root + '/' + file_name     
                    img_path_list.append(img_path)
    
    all_inf_time = []
    for img_path in img_path_list:
        im, im_data, size = read_img(img_path) 
        
        tic = time.time()
        output = sess.run(
            # output_names=['labels', 'boxes', 'scores'],
            output_names=None,
            input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}        
        )
        inf_time = time.time() - tic
        fps = float(1/inf_time)
        print('img_path: {}, inf_time: {:.4f}, FPS: {:.2f}'.format(img_path, inf_time, fps))
        all_inf_time.append(inf_time)
        
        #print(type(output))
        #print([out.shape for out in output])

        labels, boxes, scores = output
    
        draw = ImageDraw.Draw(im)  # Draw on the original image
        thrh = 0.6

        for i in range(im_data.shape[0]):

            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]

            #print(i, sum(scr > thrh))

            for b in box:
                draw.rectangle(list(b), outline='red',)
                # font = ImageFont.truetype("Arial.ttf", 15)
                draw.text((b[0], b[1]), text=str(lab[i]), fill='yellow', )

        # Save the original image with bounding boxes
        file_dir = Path(img_path).parent.parent / 'output'
        createDirectory(file_dir)
        new_file_name = os.path.basename(img_path).split('.')[0] + '_onnx'+ os.path.splitext(img_path)[1]
        new_file_path = file_dir / new_file_name
        print('new_file_path: ', new_file_path)
        print("================================================================================")
        im.save(new_file_path)
    
    avr_time = sum(all_inf_time) / len(img_path_list)
    avr_fps = float(1/avr_time)
    print('All images count: {}'.format(len(img_path_list)))
    print("Average Inferece time = {:.4f} s".format(inf_time))
    print("Average FPS = {:.2f} ".format(fps))
 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', type=str, )  # dir 
    parser.add_argument('--model', '-m', type=str, default='model.onnx')

    args = parser.parse_args()

    main(args)