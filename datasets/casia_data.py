"""
Load CASIA datasets with color, depth, and ir modalities.
Ajian Liu
2024.12.1
"""

import os, torch, random
from dassl.data.datasets import DATASET_REGISTRY
from .wrapper import FAS_MultiModal, FAS_MultiModal_VAL
from .wrapper import FAS_RGB, FAS_RGB_VAL

class DatumXY:
    """Data instance which defines the basic attributes.

    Args:
        impath_x (str): image path of fake(training) or frame1(testing).
        impath_y (str): image path of live(training) or frame2(testing).
        label (int): class label. e.g, live：0， fake：1
    """

    def __init__(self, impath_x="", impath_y="", label=-1):
        assert isinstance(impath_x, str)
        assert isinstance(impath_y, str)
        self._impath_x = impath_x
        self._impath_y = impath_y
        self._label = label

    @property
    def impath_x(self):
        return self._impath_x
    @property
    def impath_y(self):
        return self._impath_y
    @property
    def label(self):
        return self._label

def read_video(data_root, protocol, split):
    def video2list(root, txt, split):
        data_name = txt.split('@')[0]
        with open(os.path.join(root, data_name + '/protocol', txt)) as f:
            lines = f.readlines()
            f.close()
        lines_ = []
        for line in lines:
            video, label = line.strip().split(' ')
            if split == 'train':
                frame = random.choice(os.listdir(os.path.join(root, video)))
                impath = os.path.join(root, video, frame)
                lines_.append((impath, int(label)))
            else:
                frames = [os.listdir(os.path.join(root, video))[0],
                          os.listdir(os.path.join(root, video))[-1]]
                pairs = []
                for frame in frames:
                    pairs.append(os.path.join(root, video, frame))
                pairs.append(int(label))
                lines_.append(tuple(pairs))

        # data balance to 1:1
        if split == 'train':
            lives, fakes = [], []
            for line in lines_:
                impath, label = line
                if label == 0:
                    lives.append(line)
                else:
                    fakes.append(line)

            insert = len(fakes) - len(lives)
            if insert >= 0:
                for _ in range(insert):
                    lives.append(random.choice(lives))
            else:
                for _ in range(-insert):
                    fakes.append(random.choice(fakes))

            assert len(lives) == len(fakes)
            return lives, fakes
        else:
            return lines_

    ##########
    items = []
    if split == 'train':
        lives_list, fakes_list = video2list(data_root, protocol + '_video_' + split + '.txt', split)
        for i in range(len(fakes_list)):
            item = DatumXY(
                impath_x=fakes_list[i][0],
                impath_y=lives_list[i][0],
                label=fakes_list[i][1]
            )
            items.append(item)
        print('Load {} {}={}'.format(protocol, split, len(lives_list)))
        return items
    else:
        impath_label_list = video2list(data_root, protocol + '_video_' + split + '.txt', split)
        for impath1, impath2, label in impath_label_list:
            item = DatumXY(
                impath_x=impath1,
                impath_y=impath2,
                label=label
            )
            items.append(item)
        print('Load {} {}={}'.format(protocol, split, len(impath_label_list)))
        return items


def read_image(data_root, protocol, split):
    def image2list(root, txt, split):
        data_name = txt.split('@')[0]
        with open(os.path.join(root, data_name + '/protocol', txt)) as f:
            lines = f.readlines()
            f.close()
        lines_ = []
        for line in lines:
            image, label = line.strip().split(' ')
            impath = os.path.join(root, image)
            lines_.append((impath, int(label)))

        # data balance to 1:1
        if split == 'train':
            lives, fakes = [], []
            for line in lines_:
                impath, label = line
                if label == 0:
                    lives.append(line)
                else:
                    fakes.append(line)
            insert = len(fakes) - len(lives)

            if insert >= 0:
                for _ in range(insert):
                    lives.append(random.choice(lives))
            else:
                for _ in range(-insert):
                    fakes.append(random.choice(fakes))

            assert len(lives) == len(fakes)
            return lives, fakes
        else:
            return lines_

    ##########
    items = []
    if split == 'train':
        lives_list, fakes_list = image2list(data_root, protocol + '_image_' + split + '.txt', split)
        for i in range(len(fakes_list)):
            item = DatumXY(
                impath_x=fakes_list[i][0],
                impath_y=lives_list[i][0],
                label=fakes_list[i][1]
            )
            items.append(item)
        print('Load {} {}={}'.format(protocol, split, len(lives_list)))
        return items
    else:
        impath_label_list = image2list(data_root, protocol + '_image_' + split + '.txt', split)
        for impath, label in impath_label_list:
            item = DatumXY(
                impath_x=impath,
                impath_y=impath,
                label=label
            )
            items.append(item)
        print('Load {} {}={}'.format(protocol, split, len(impath_label_list)))
        return items


def build_dataset(data_root, protocol, is_video):
    if is_video:
        data_train = read_video(data_root, protocol, split='train')
        data_dev = read_video(data_root, protocol, split='dev')
        data_test = read_video(data_root, protocol, split='test')
    else:
        data_train = read_image(data_root, protocol, split='train')
        data_dev = read_image(data_root, protocol, split='dev')
        data_test = read_image(data_root, protocol, split='test')
    return data_train, data_dev, data_test

@DATASET_REGISTRY.register()
class CDI_DATA:
    def __init__(self, cfg):
        modals = ['color', 'depth', 'ir']
        train, dev, test = build_dataset(cfg.DATASET.ROOT, cfg.DATASET.PROTOCOL, cfg.DATASET.IS_VIDEO)

        # Build data loader
        train_loader = torch.utils.data.DataLoader(
            FAS_MultiModal(
                    data_source=train,
                    image_size=cfg.INPUT.SIZE[0],
                    modals=modals,
                    preprocess=cfg.DATASET.PREPROCESS),
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    shuffle=True,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    drop_last=True,
                    pin_memory=False
                    )
        dev_loader = torch.utils.data.DataLoader(
            FAS_MultiModal_VAL(
                    data_source=dev,
                    image_size=cfg.INPUT.SIZE[0],
                    modals=modals,
                    preprocess='resize'),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=False
                    )

        test_loader = torch.utils.data.DataLoader(
            FAS_MultiModal_VAL(
                    data_source=test,
                    image_size=cfg.INPUT.SIZE[0],
                    modals=modals,
                    preprocess='resize'),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=False
                    )

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.lab2cname = {0: 'live', 1: 'fake'}
        self.classnames = ['live', 'fake']
        self.modalnames = ['color', 'depth', 'ir']
        self.templates = [
            'This is an example of a {} face',
            'This is a {} face',
            'This is how a {} face looks like',
            'A photo of a {} face',
            'Is not this a {} face ?',
            'A printout shown to be a {} face'
        ]


@DATASET_REGISTRY.register()
class C_DATA:
    def __init__(self, cfg):
        train, dev, test = build_dataset(cfg.DATASET.ROOT, cfg.DATASET.PROTOCOL, cfg.DATASET.IS_VIDEO)

        # Build data loader
        train_loader = torch.utils.data.DataLoader(
            FAS_RGB(
                    data_source=train,
                    image_size=cfg.INPUT.SIZE[0],
                    preprocess=cfg.DATASET.PREPROCESS,
                    task='intra'),
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    shuffle=True,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    drop_last=True,
                    pin_memory=False
                    )
        dev_loader = torch.utils.data.DataLoader(
            FAS_RGB_VAL(
                    data_source=dev,
                    image_size=cfg.INPUT.SIZE[0],
                    preprocess='resize'),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=False
                    )

        test_loader = torch.utils.data.DataLoader(
            FAS_RGB_VAL(
                    data_source=test,
                    image_size=cfg.INPUT.SIZE[0],
                    preprocess='resize'),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=False
                    )

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.lab2cname = {0: 'live', 1: 'fake'}
        self.classnames = ['live', 'fake']
        self.templates = [
            'This is an example of a {} face',
            'This is a {} face',
            'This is how a {} face looks like',
            'A photo of a {} face',
            'Is not this a {} face ?',
            'A printout shown to be a {} face'
        ]



