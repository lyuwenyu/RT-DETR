'''by lyuwenyu
'''

import os
import glob

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T 
import torchvision.transforms.functional as F 


class ToTensor(T.ToTensor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic 
        return super().__call__(pic)


class PadToSize(T.Pad):
    def __init__(self, size, fill=0, padding_mode='constant'):
        super().__init__(0, fill, padding_mode)
        self.size = size
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        w, h = F.get_image_size(img)
        padding = (0, 0, self.size[0] - w, self.size[1] - h)
        return F.pad(img, padding, self.fill, self.padding_mode)


class Dataset(data.Dataset):
    def __init__(self, img_dir: str='', preprocess: T.Compose=None, device='cuda:0') -> None:
        super().__init__()

        self.device = device
        self.size = 640

        self.im_path_list = list(glob.glob(os.path.join(img_dir, '*.jpg')))

        if preprocess is None: 
            self.preprocess = T.Compose([
                    T.Resize(size=639, max_size=640),
                    PadToSize(size=(640, 640), fill=114),
                    ToTensor(),
                    T.ConvertImageDtype(torch.float),
            ])
        else:
            self.preprocess = preprocess

    def __len__(self, ):
        return len(self.im_path_list)


    def __getitem__(self, index):
        # im = Image.open(self.img_path_list[index]).convert('RGB')
        im = torchvision.io.read_file(self.im_path_list[index])
        im = torchvision.io.decode_jpeg(im, mode=torchvision.io.ImageReadMode.RGB, device=self.device)
        _, h, w = im.shape # c,h,w

        im = self.preprocess(im)

        blob = {
            'image': im, 
            'im_shape': torch.tensor([self.size, self.size]).to(im.device),
            'scale_factor': torch.tensor([h / self.size, w / self.size]).to(im.device),
            'orig_size': torch.tensor([w, h]).to(im.device),
        }

        return blob


    @staticmethod
    def post_process():
        pass

    @staticmethod
    def collate_fn():
        pass
