"""
Function: DatasetWrapper with transforms
Author: Ajian Liu
Date: 2024/12/1
"""

import random, torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from util.utils_FAS import check_if_exist
import torchvision.transforms.functional as F

def get_params(resize, crop_size, degrees=180):
    w, h = resize, resize
    x = random.randint(0, np.maximum(0, w - crop_size))
    y = random.randint(0, np.maximum(0, h - crop_size))
    flip = random.random() > 0.5
    rotate = random.random() > 0.5
    angle = random.uniform(-degrees, degrees)
    ColorJitter = random.random() > 0.5
    brightness, contrast, saturation, hue = \
        random.random()/2.0, random.random()/2.0, random.random()/2.0, random.random()/2.0
    return {'crop_pos': (x, y), 'flip': flip, 'rotate': rotate, 'angle': angle, 'ColorJitter':ColorJitter,
            'brightness': brightness, 'contrast': contrast, 'saturation': saturation, 'hue': hue}
def _crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img
def _flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
def _rotate(img, angle):
    return img.rotate(angle)
def _get_ColorJitter(brightness, contrast, saturation, hue):
    """Get a randomized transform to be applied on image.
    Returns:
        Transform which randomly adjusts brightness, contrast and saturation in a random order.
    """
    transforms_list = []
    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))
    random.shuffle(transforms_list)
    return transforms_list

def get_transform(preprocess, params, load_size=300, crop_size=256):
    transform_list = []
    if 'resize' in preprocess:transform_list.append(transforms.Resize([load_size, load_size]))
    if 'crop' in preprocess:
        if params is None:transform_list.append(transforms.RandomCrop(crop_size))
        else:transform_list.append(transforms.Lambda(lambda img: _crop(img, params['crop_pos'], crop_size)))
    else:
        osize = [crop_size, crop_size]
        transform_list.append(transforms.Resize(osize))
    if params is not None:
        if ('flip' in preprocess) and (params['flip']):transform_list.append(transforms.Lambda(lambda img: _flip(img)))
        if ('rotate' in preprocess) and (params['rotate']):
            transform_list.append(transforms.Lambda(lambda img: _rotate(img, params['angle'])))
        if ('ColorJitter' in preprocess) and (params['ColorJitter']):
            ColorJitter = transforms.ColorJitter(brightness=params['brightness'], contrast=params['contrast'], saturation=params['saturation'], hue=params['hue'])
            transform_list += _get_ColorJitter(ColorJitter.brightness, ColorJitter.contrast, ColorJitter.saturation, ColorJitter.hue)
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
    return transforms.Compose(transform_list)


class FAS_RGB(data.Dataset):
    def __init__(self, data_source, image_size, preprocess, task=''):
        super(FAS_RGB, self).__init__()
        self.data_source = data_source
        self.load_size = 300
        self.rotate = 180
        self.depth_size = 32
        self.image_size = image_size
        self.preprocess = preprocess
        self.task = task

    def __len__(self):
        return len(self.data_source)

    def get_data(self, path, input_transform):
        img = Image.open(path).convert('RGB')
        img_r = input_transform(img)
        return img_r

    def __getitem__(self, index):
        item = self.data_source[index]

        t_params = get_params(self.load_size, crop_size=self.image_size, degrees=self.rotate)
        input_transform = get_transform(self.preprocess, params=t_params, crop_size=self.image_size)
        x_b = torch.zeros(1, self.depth_size, self.depth_size)
        y_b = torch.ones(1, self.depth_size, self.depth_size)

        if self.task == 'dg':
            x_path, x_label, x_domain, y_path, y_label, y_domain = \
                item.impath_x, 1, item.domain, item.impath_y, 0, item.domain
            x_r = self.get_data(x_path, input_transform)
            y_r = self.get_data(y_path, input_transform)
            return {'X_R': x_r, 'X_L': x_label, 'X_D': x_domain,
                    'Y_R': y_r, 'Y_L': y_label, 'Y_D': y_domain}
        elif self.task == 'intra':
            x_path, x_label, y_path, y_label = item.impath_x, 1, item.impath_y, 0
            x_r = self.get_data(x_path, input_transform)
            y_r = self.get_data(y_path, input_transform)
            return {'X_R': x_r, 'X_L': x_label, 'X_D': x_b,
                    'Y_R': y_r, 'Y_L': y_label, 'Y_D': y_b}
        else:
            x_path, x_label, y_path, y_label = item.impath_x, 1, item.impath_y, 0
            x_r = self.get_data(x_path, input_transform)
            y_r = self.get_data(y_path, input_transform)
            return {'X_R': x_r, 'X_B': x_b, 'X_P': x_path, 'X_L': x_label,
                    'Y_R': y_r, 'Y_B': y_b, 'Y_P': y_path, 'Y_L': y_label}


class FAS_RGB_VAL(data.Dataset):
    def __init__(self, data_source, image_size, preprocess):
        super(FAS_RGB_VAL, self).__init__()
        self.data_source = data_source
        self.load_size = 300
        self.rotate = 0
        self.image_size = image_size
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_source)
    def __getitem__(self, index):
        item = self.data_source[index]

        t_params = get_params(self.load_size, crop_size=self.image_size, degrees=self.rotate)
        input_transform = get_transform(self.preprocess, params=t_params, crop_size=self.image_size)
        path1, path2, label = item.impath_x, item.impath_y, item.label
        if label != 0:label = 1
        frame1 = input_transform(Image.open(path1).convert('RGB'))
        frame2 = input_transform(Image.open(path2).convert('RGB'))
        return {'frame1': frame1, 'frame2': frame2, 'label': label, 'path': path1}


def replace(image_name, modal_1, modal_2):
    image_name = image_name.replace(modal_1, modal_2)
    assert modal_2 in image_name
    if not check_if_exist(image_name):
        print(image_name)
        exit(0)
    return image_name
class FAS_MultiModal(data.Dataset):
    def __init__(self, data_source, image_size, modals, preprocess):
        super(FAS_MultiModal, self).__init__()
        self.data_source = data_source
        self.load_size = 300
        self.rotate = 180
        self.depth_size = 32
        self.image_size = image_size
        self.modals = modals
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_source)

    def get_data(self, path, input_transform):
        ### RGB & Depth & IR
        rgb = Image.open(path).convert('RGB')
        depth = Image.open(replace(path, self.modals[0], self.modals[1])).convert('RGB')
        ir = Image.open(replace(path, self.modals[0], self.modals[2])).convert('RGB')
        rgb = input_transform(rgb)
        depth = input_transform(depth)
        ir = input_transform(ir)
        image_3 = torch.cat([rgb, depth, ir], dim=0)
        return image_3

    def __getitem__(self, index):
        item = self.data_source[index]
        x_path, x_label, y_path, y_label = item.impath_x, 1, item.impath_y, 0
        t_params = get_params(self.load_size, crop_size=self.image_size, degrees=self.rotate)
        input_transform = get_transform(self.preprocess, params=t_params, crop_size=self.image_size)
        x_3 = self.get_data(x_path, input_transform)
        y_3 = self.get_data(y_path, input_transform)

        x_b = torch.zeros(1, self.depth_size, self.depth_size)
        y_b = torch.ones(1, self.depth_size, self.depth_size)

        return {'X_3': x_3, 'X_B': x_b, 'X_P': x_path, 'X_L': x_label,
                'Y_3': y_3, 'Y_B': y_b, 'Y_P': y_path, 'Y_L': y_label}


class FAS_MultiModal_VAL(data.Dataset):
    def __init__(self, data_source, image_size, modals, preprocess):
        super(FAS_MultiModal_VAL, self).__init__()
        self.data_source = data_source
        self.load_size = 300
        self.rotate = 180
        self.modals = modals
        self.image_size = image_size
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_source)

    def get_data(self, path, input_transform):
        ### RGB & Depth & IR
        rgb = Image.open(path).convert('RGB')
        depth = Image.open(replace(path, self.modals[0], self.modals[1])).convert('RGB')
        ir = Image.open(replace(path, self.modals[0], self.modals[2])).convert('RGB')
        rgb = input_transform(rgb)
        depth = input_transform(depth)
        ir = input_transform(ir)
        image_3 = torch.cat([rgb, depth, ir], dim=0)
        return image_3

    def __getitem__(self, index):
        item = self.data_source[index]
        path1, path2, label = item.impath_x, item.impath_y, item.label
        ### apply the same transform to both input and depth
        t_params = get_params(self.load_size, crop_size=self.image_size, degrees=self.rotate)
        input_transform = get_transform(self.preprocess, params=t_params, crop_size=self.image_size)
        frame1 = self.get_data(path1, input_transform)
        frame2 = self.get_data(path2, input_transform)
        return {'frame1': frame1, 'frame2': frame2, 'label': label, 'path': path1}


