"""
Load domain generalized datasetes, i.e., CASIA_FASD, MSU_MFSD, ReplayAttack, OULU_NPU, etc.
Ajian Liu
2024.12.10
"""

import torch, os, random
import util.utils_FAS as utils
from dassl.data.datasets import DATASET_REGISTRY
from .samplers import build_sampler
from .wrapper import FAS_RGB, FAS_RGB_VAL

class DatumXY:
    """Data instance which defines the basic attributes.
    Args:
        impath_x (str): image path of fake.
        impath_y (str): image path of live.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath_x="", impath_y="", label=-1, domain=-1, classname="", video=""):
        assert isinstance(impath_x, str)
        assert isinstance(impath_y, str)
        self._impath_x = impath_x
        self._impath_y = impath_y
        self._label = label
        self._domain = domain
        self._classname = classname
        self._video = video

    @property
    def impath_x(self):
        return self._impath_x
    @property
    def impath_y(self):
        return self._impath_y
    @property
    def label(self):
        return self._label
    @property
    def domain(self):
        return self._domain
    @property
    def classname(self):
        return self._classname
    @property
    def video(self):
        return self._video


def video2list(root, txt, folder, split):
    # get data from txt
    with open(os.path.join(root, 'dg_protocol', txt)) as f:
        lines = f.readlines()
        f.close()
    lines_ = []
    for line in lines:
        video, label = line.strip().split(' ')
        if split == 'train':
            if not utils.check_if_exist(os.path.join(root, video, folder)): folder = 'color'
            if not utils.check_if_exist(os.path.join(root, video, folder)): folder = ''
            frame = random.choice(os.listdir(os.path.join(root, video, folder)))
            impath = os.path.join(root, video, folder, frame)
            lines_.append((impath, int(label)))
        else:
            if not utils.check_if_exist(os.path.join(root, video, folder)): folder = 'color'
            if not utils.check_if_exist(os.path.join(root, video, folder)): folder = ''
            frames = [os.listdir(os.path.join(root, video, folder))[0],
                      os.listdir(os.path.join(root, video, folder))[-1]]
            pairs = []
            for frame in frames:
                impath = os.path.join(root, video, folder, frame)
                if not utils.check_if_exist(impath): impath = impath.replace('.jpg', '.png')
                pairs.append(impath)
            pairs.append(video)
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
        if insert > 0:
            for _ in range(insert):
                lives.append(random.choice(lives))
        else:
            for _ in range(-insert):
                fakes.append(random.choice(fakes))

        assert len(lives) == len(fakes)
        return lives, fakes
    else:
        return lines_

def read_video(data_root, input_domains, folder, split):
    items = []
    if split == 'train':
        for domain, dname in enumerate(input_domains):
            lives_list, fakes_list = video2list(data_root, dname + '_video_' + split + '.txt', folder, split)
            for i in range(len(fakes_list)):
                item = DatumXY(
                    impath_x=fakes_list[i][0],
                    impath_y=lives_list[i][0],
                    domain=domain
                )
                items.append(item)
            print('Load {} {}={}'.format(dname, split, len(lives_list)))
        return items
        ### Train with additional CelebA_Spoof dataset
        txt = '/mnt/sdh/LAJ_data/ICMO/images/CelebA_Spoof/CelebA_Spoof_image_fakelive.txt'
        with open(txt) as f:
            lines = f.readlines()
        f.close()
        for line in lines:
            fakepath, livepath = line.strip().split(' ')
            item = DatumXY(
                impath_x=fakepath,
                impath_y=livepath,
                domain=-1
                    )
            items.append(item)
        print('Load {} {}={}'.format('CelebA_Spoof_image_fakelive', split, len(lines)))
        return items
    else:
        for domain, dname in enumerate(input_domains):
            impath_label_list = video2list(data_root, dname + '_video_' + split + '.txt', folder, split)
            for impath1, impath2, video, label in impath_label_list:
                item = DatumXY(
                    impath_x=impath1,
                    impath_y=impath2,
                    label=label,
                    video=video
                )
                items.append(item)
            print('Load {} {}={}'.format(dname, split, len(impath_label_list)))
        return items
        #### Test with another 3DMask dataset
        txt = '/mnt/sdh/LAJ_data/DKMH/splits_text/3DMask_video_test.txt'
        with open(txt) as f:
            lines = f.readlines()
        f.close()
        for line in lines:
            impath, label = line.strip().split(' ')
            item = DatumXY(
                impath_x='/mnt/sdh/LAJ_data/DKMH/images/' + impath + '/05.png',
                impath_y='/mnt/sdh/LAJ_data/DKMH/images/' + impath + '/05.png',
                label=int(label),
                video='-1'
            )
            items.append(item)
        print('Load {} {}={}'.format('CelebA_Spoof_image_test', split, len(lines)))
        return items

def build_dataset(data_root, protocol, folder='crop'):
    train_name, test_name = protocol.split('@')
    data_train = read_video(data_root, train_name.split('-'), folder, split='train')
    data_dev = read_video(data_root, test_name.split('-'), folder, split='dev')
    data_test = read_video(data_root, test_name.split('-'), folder, split='test')
    return data_train, data_dev, data_test

# RandomSampler SequentialSampler RandomDomainSampler SeqDomainSampler RandomClassSampler
@DATASET_REGISTRY.register()
class DG_DATA:
    def __init__(self, cfg):
        sampler_type = "SeqDomainSampler"
        train, dev, test = build_dataset(cfg.DATASET.ROOT, cfg.DATASET.PROTOCOL)
        # Build sampler
        sampler = build_sampler(
                sampler_type,
                data_source=train,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS
                )

        # Build data loader
        train_loader = torch.utils.data.DataLoader(
            FAS_RGB(
                data_source=train,
                image_size=cfg.INPUT.SIZE[0],
                preprocess=cfg.DATASET.PREPROCESS,
                task='dg'),
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                sampler=sampler,
                # shuffle=True,
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
                # sampler=sampler,
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
                # sampler=sampler,
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
        self.domainnames = ['0', '1', '2']
        self.templates = [
            'This is an example of a {} face',
            'This is a {} face',
            'This is how a {} face looks like',
            'A photo of a {} face',
            'Is not this a {} face ?',
            'A printout shown to be a {} face'
        ]


