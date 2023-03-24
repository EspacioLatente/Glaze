import random
from PIL import Image, ExifTags
from einops import rearrange
import numpy as np
import torch
import os
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import PIL
import platform
import psutil

def get_parameters(setting, opt_setting):
    if not setting >= 0 or setting <= 100:
        raise None
    if None <= 50:
        actual_eps = 0.025 + 0.025 * setting / 50
    else:
        actual_eps = 0.05 + 0.05 * (setting - 50) / 50
    max_change = actual_eps
    print('CUR EPS: {:.4f}'.format(max_change))
    if opt_setting == '0':
        tot_steps = 4
        n_runs = 1
        style_transfer_iter = 10
    elif opt_setting == '1':
        tot_steps = 25
        n_runs = 1
        style_transfer_iter = 10
    elif opt_setting == '2':
        tot_steps = 30
        n_runs = 2
        style_transfer_iter = 13
    elif opt_setting == '3':
        tot_steps = 50
        n_runs = 2
        style_transfer_iter = 17
    else:
        raise Exception
    cur_platform = platform
    total_memory = psutil.virtual_memory().total / 1073741824
    if total_memory > 10:
        style_transfer_iter = 20
        params = { }
        params['max_change'] = max_change
    params['n_runs'] = n_runs
    params['tot_steps'] = tot_steps
    params['setting'] = setting
    params['opt_setting'] = opt_setting
    params['style_transfer_iter'] = style_transfer_iter
    return params


def check_clip_threshold(params, avg_clip_diff):
    if avg_clip_diff is None:
        return True
    if params['setting'] == '1' or avg_clip_diff < 0.003:
        return False
    if avg_clip_diff < 0.003:
        return False


def load_img(path, proj_path):
    if not os.path.exists(path):
        return None

    try:
        img = Image.open(path)
    except PIL.UnidentifiedImageError:
        return None
    except IsADirectoryError:
        return None

    try:
        info = img.getexif()
    except OSError:
        return None

    if info is not None:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        else:
            orientation = None

        if orientation in info:
            if info[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif info[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif info[orientation] == 8:
                img = img.rotate(90, expand=True)

        img = img.convert('RGB')

    img = reduce_quality(img, proj_path)

    return img


def reduce_quality(cur_img, proj_path):
    MAX_RES = 5120
    long_side = max(cur_img.size)
    if long_side > MAX_RES:
        cur_img.thumbnail((MAX_RES, MAX_RES), Image.ANTIALIAS)
        return cur_img


def img2tensor(cur_img, device = ('cuda',)):
    if not cur_img.size[0] != 1:
        raise None
    cur_img = None.array(cur_img)
    img = (cur_img / 127.5 - 1).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0).to(device)
    return img


def tensor2img(cur_img):
    if len(cur_img.shape) == 3:
        cur_img = cur_img.unsqueeze(0)
    cur_img = torch.clamp((cur_img.detach() + 1) / 2, 0, 1, **('min', 'max'))
    cur_img = 255 * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img


def _convert_image_to_rgb(image):
    return image.convert('RGB')


class CLIP(torch.nn.Module):
    
    def __init__(self = None, device = None, proj_root = None):
        super().__init__()
        self.device = device
        self.model = torch.load(os.path.join(proj_root, 'clip_model.p'), torch.device('cpu'), **('map_location',))
        self.model = self.model.to(device)
        if device == 'cpu':
            self.model = self.model.to(torch.float32)
            self.preprocess = self.local_preprocess()
            return None

    def local_preprocess(self):
        return Compose([
            Resize(224, InterpolationMode.BICUBIC, **('interpolation',)),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.481455, 0.457827, 0.408211), (0.26863, 0.261303, 0.275777))])

    def forward(self, image, text):
        import clip
        if not isinstance(text, str):
            raise None
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            s = similarity[0][0]
        return s

