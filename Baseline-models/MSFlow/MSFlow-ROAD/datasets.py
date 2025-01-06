import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

__all__ = ('MVTecDataset', )

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path # './data/MVTec'
        self.class_name = c.class_name  # './data/MVTec'
        self.is_train = is_train # True
        self.input_size = c.input_size  # (512, 512)
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()  #？ 都是list(209)  x放路径，y放的全是0，mask放的全是None
        # set transforms
        if is_train:
            self.transform_x = T.Compose([  #from torchvision import transforms as T
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase) #'./data/MVTec\\bottle\\train'
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth') #'./data/MVTec\\bottle\\ground_truth'

        img_types = sorted(os.listdir(img_dir)) # ['good']


        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type) #'./data/MVTec\\bottle\\train\\good'
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)]) # 长度209 #'./data/MVTec\\bottle\\train\\good'
            x.extend(img_fpath_list) # list(209) 存储路径 ['./data/MVTec\\bottle\\train\\good\\000.png', './data/MVTec\\bottle\\train\\good\\001.png', './data/MVTec\\bottle\\train\\good\\002.png', './data/MVTec\\bottle\\train\\good\\003.png', './data/MVTec\\bottle\\train\\good\\004.png', './data/MVTec\\bottle\\train\\good\\005.png', './data/MVTec\\bottle\\train\\good\\006.png', './data/MVTec\\bottle\\train\\good\\007.png', './data/MVTec\\bottle\\train\\good\\008.png', './data/MVTec\\bottle\\train\\good\\009.png', './data/MVTec\\bottle\\train\\good\\010.png', './data/MVTec\\bottle\\train\\good\\011.png', './data/MVTec\\bottle\\train\\good\\012.png', './data/MVTec\\bottle\\train\\good\\013.png', './data/MVTec\\bottle\\train\\good\\014.png', './data/MVTec\\bottle\\train\\good\\015.png', './data/MVTec\\bottle\\train\\good\\016.png', './data/MVTec\\bottle\\train\\good\\017.png', './data/MVTec\\bottle\\train\\good\\018.png', './data/MVTec\\bottle\\train\\good\\019.png', './data/MVTec\\bottle\\train\\good\\020.png', './data/MVTec\\bottle\\train\\good\\021.png', './data/MVTec\\bottle\\train\\good\\022.png', './data/MVTec\\bottle\\train\\good\\023.png', './data/MVTec\\bottle\\train\\good\\024.png', './data/MVTec\\bottle\\train\\good\\025.png', './data/MVTec\\bottle\\train\\good\\026.png', './data/MVTec\\bottle\\train\\good\\027.png', './data/MVTec\\bottle\\train\\good\\028.png', './data/MVTec\\bottle\\train\\good\\029.png', './data/MVTec\\bottle\\train\\good\\030.png', './data/MVTec\\bottle\\train\\good\\031.png', './data/MVTec\\bottle\\train\\good\\032.png', './data/MVTec\\bottle\\train\\good\\033.png', './data/MVTec\\bottle\\train\\good\\034.png', './data/MVTec\\bottle\\train\\good\\035.png', './data/MVTec\\bottle\\train\\good\\036.png', './data/MVTec\\bottle\\train\\good\\037.png', './data/MVTec\\bottle\\train\\good\\038.png', './data/MVTec\\bottle\\train\\good\\039.png', './data/MVTec\\bottle\\train\\good\\040.png', './data/MVTec\\bottle\\train\\good\\041.png', './data/MVTec\\bottle\\train\\good\\042.png', './data/MVTec\\bottle\\train\\good\\043.png', './data/MVTec\\bottle\\train\\good\\044.png', './data/MVTec\\bottle\\train\\good\\045.png', './data/MVTec\\bottle\\train\\good\\046.png', './data/MVTec\\bottle\\train\\good\\047.png', './data/MVTec\\bottle\\train\\good\\048.png', './data/MVTec\\bottle\\train\\good\\049.png', './data/MVTec\\bottle\\train\\good\\050.png', './data/MVTec\\bottle\\train\\good\\051.png', './data/MVTec\\bottle\\train\\good\\052.png', './data/MVTec\\bottle\\train\\good\\053.png', './data/MVTec\\bottle\\train\\good\\054.png', './data/MVTec\\bottle\\train\\good\\055.png', './data/MVTec\\bottle\\train\\good\\056.png', './data/MVTec\\bottle\\train\\good\\057.png', './data/MVTec\\bottle\\train\\good\\058.png', './data/MVTec\\bottle\\train\\good\\059.png', './data/MVTec\\bottle\\train\\good\\060.png', './data/MVTec\\bottle\\train\\good\\061.png', './data/MVTec\\bottle\\train\\good\\062.png', './data/MVTec\\bottle\\train\\good\\063.png', './data/MVTec\\bottle\\train\\good\\064.png', './data/MVTec\\bottle\\train\\good\\065.png', './data/MVTec\\bottle\\train\\good\\066.png', './data/MVTec\\bottle\\train\\good\\067.png', './data/MVTec\\bottle\\train\\good\\068.png', './data/MVTec\\bottle\\train\\good\\069.png', './data/MVTec\\bottle\\train\\good\\070.png', './data/MVTec\\bottle\\train\\good\\071.png', './data/MVTec\\bottle\\train\\good\\072.png', './data/MVTec\\bottle\\train\\good\\073.png', './data/MVTec\\bottle\\train\\good\\074.png', './data/MVTec\\bottle\\train\\good\\075.png', './data/MVTec\\bottle\\train\\good\\076.png', './data/MVTec\\bottle\\train\\good\\077.png', './data/MVTec\\bottle\\train\\good\\078.png', './data/MVTec\\bottle\\train\\good\\079.png', './data/MVTec\\bottle\\train\\good\\080.png', './data/MVTec\\bottle\\train\\good\\081.png', './data/MVTec\\bottle\\train\\good\\082.png', './data/MVTec\\bottle\\train\\good\\083.png', './data/MVTec\\bottle\\train\\good\\084.png', './data/MVTec\\bottle\\train\\good\\085.png', './data/MVTec\\bottle\\train\\good\\086.png', './data/MVTec\\bottle\\train\\good\\087.png', './data/MVTec\\bottle\\train\\good\\088.png', './data/MVTec\\bottle\\train\\good\\089.png', './data/MVTec\\bottle\\train\\good\\090.png', './data/MVTec\\bottle\\train\\good\\091.png', './data/MVTec\\bottle\\train\\good\\092.png', './data/MVTec\\bottle\\train\\good\\093.png', './data/MVTec\\bottle\\train\\good\\094.png', './data/MVTec\\bottle\\train\\good\\095.png', './data/MVTec\\bottle\\train\\good\\096.png', './data/MVTec\\bottle\\train\\good\\097.png', './data/MVTec\\bottle\\train\\good\\098.png', './data/MVTec\\bottle\\train\\good\\099.png', './data/MVTec\\bottle\\train\\good\\100.png', './data/MVTec\\bottle\\train\\good\\101.png', './data/MVTec\\bottle\\train\\good\\102.png', './data/MVTec\\bottle\\train\\good\\103.png', './data/MVTec\\bottle\\train\\good\\104.png', './data/MVTec\\bottle\\train\\good\\105.png', './data/MVTec\\bottle\\train\\good\\106.png', './data/MVTec\\bottle\\train\\good\\107.png', './data/MVTec\\bottle\\train\\good\\108.png', './data/MVTec\\bottle\\train\\good\\109.png', './data/MVTec\\bottle\\train\\good\\110.png', './data/MVTec\\bottle\\train\\good\\111.png', './data/MVTec\\bottle\\train\\good\\112.png', './data/MVTec\\bottle\\train\\good\\113.png', './data/MVTec\\bottle\\train\\good\\114.png', './data/MVTec\\bottle\\train\\good\\115.png', './data/MVTec\\bottle\\train\\good\\116.png', './data/MVTec\\bottle\\train\\good\\117.png', './data/MVTec\\bottle\\train\\good\\118.png', './data/MVTec\\bottle\\train\\good\\119.png', './data/MVTec\\bottle\\train\\good\\120.png', './data/MVTec\\bottle\\train\\good\\121.png', './data/MVTec\\bottle\\train\\good\\122.png', './data/MVTec\\bottle\\train\\good\\123.png', './data/MVTec\\bottle\\train\\good\\124.png', './data/MVTec\\bottle\\train\\good\\125.png', './data/MVTec\\bottle\\train\\good\\126.png', './data/MVTec\\bottle\\train\\good\\127.png', './data/MVTec\\bottle\\train\\good\\128.png', './data/MVTec\\bottle\\train\\good\\129.png', './data/MVTec\\bottle\\train\\good\\130.png', './data/MVTec\\bottle\\train\\good\\131.png', './data/MVTec\\bottle\\train\\good\\132.png', './data/MVTec\\bottle\\train\\good\\133.png', './data/MVTec\\bottle\\train\\good\\134.png', './data/MVTec\\bottle\\train\\good\\135.png', './data/MVTec\\bottle\\train\\good\\136.png', './data/MVTec\\bottle\\train\\good\\137.png', './data/MVTec\\bottle\\train\\good\\138.png', './data/MVTec\\bottle\\train\\good\\139.png', './data/MVTec\\bottle\\train\\good\\140.png', './data/MVTec\\bottle\\train\\good\\141.png', './data/MVTec\\bottle\\train\\good\\142.png', './data/MVTec\\bottle\\train\\good\\143.png', './data/MVTec\\bottle\\train\\good\\144.png', './data/MVTec\\bottle\\train\\good\\145.png', './data/MVTec\\bottle\\train\\good\\146.png', './data/MVTec\\bottle\\train\\good\\147.png', './data/MVTec\\bottle\\train\\good\\148.png', './data/MVTec\\bottle\\train\\good\\149.png', './data/MVTec\\bottle\\train\\good\\150.png', './data/MVTec\\bottle\\train\\good\\151.png', './data/MVTec\\bottle\\train\\good\\152.png', './data/MVTec\\bottle\\train\\good\\153.png', './data/MVTec\\bottle\\train\\good\\154.png', './data/MVTec\\bottle\\train\\good\\155.png', './data/MVTec\\bottle\\train\\good\\156.png', './data/MVTec\\bottle\\train\\good\\157.png', './data/MVTec\\bottle\\train\\good\\158.png', './data/MVTec\\bottle\\train\\good\\159.png', './data/MVTec\\bottle\\train\\good\\160.png', './data/MVTec\\bottle\\train\\good\\161.png', './data/MVTec\\bottle\\train\\good\\162.png', './data/MVTec\\bottle\\train\\good\\163.png', './data/MVTec\\bottle\\train\\good\\164.png', './data/MVTec\\bottle\\train\\good\\165.png', './data/MVTec\\bottle\\train\\good\\166.png', './data/MVTec\\bottle\\train\\good\\167.png', './data/MVTec\\bottle\\train\\good\\168.png', './data/MVTec\\bottle\\train\\good\\169.png', './data/MVTec\\bottle\\train\\good\\170.png', './data/MVTec\\bottle\\train\\good\\171.png', './data/MVTec\\bottle\\train\\good\\172.png', './data/MVTec\\bottle\\train\\good\\173.png', './data/MVTec\\bottle\\train\\good\\174.png', './data/MVTec\\bottle\\train\\good\\175.png', './data/MVTec\\bottle\\train\\good\\176.png', './data/MVTec\\bottle\\train\\good\\177.png', './data/MVTec\\bottle\\train\\good\\178.png', './data/MVTec\\bottle\\train\\good\\179.png', './data/MVTec\\bottle\\train\\good\\180.png', './data/MVTec\\bottle\\train\\good\\181.png', './data/MVTec\\bottle\\train\\good\\182.png', './data/MVTec\\bottle\\train\\good\\183.png', './data/MVTec\\bottle\\train\\good\\184.png', './data/MVTec\\bottle\\train\\good\\185.png', './data/MVTec\\bottle\\train\\good\\186.png', './data/MVTec\\bottle\\train\\good\\187.png', './data/MVTec\\bottle\\train\\good\\188.png', './data/MVTec\\bottle\\train\\good\\189.png', './data/MVTec\\bottle\\train\\good\\190.png', './data/MVTec\\bottle\\train\\good\\191.png', './data/MVTec\\bottle\\train\\good\\192.png', './data/MVTec\\bottle\\train\\good\\193.png', './data/MVTec\\bottle\\train\\good\\194.png', './data/MVTec\\bottle\\train\\good\\195.png', './data/MVTec\\bottle\\train\\good\\196.png', './data/MVTec\\bottle\\train\\good\\197.png', './data/MVTec\\bottle\\train\\good\\198.png', './data/MVTec\\bottle\\train\\good\\199.png', './data/MVTec\\bottle\\train\\good\\200.png', './data/MVTec\\bottle\\train\\good\\201.png', './data/MVTec\\bottle\\train\\good\\202.png', './data/MVTec\\bottle\\train\\good\\203.png', './data/MVTec\\bottle\\train\\good\\204.png', './data/MVTec\\bottle\\train\\good\\205.png', './data/MVTec\\bottle\\train\\good\\206.png', './data/MVTec\\bottle\\train\\good\\207.png', './data/MVTec\\bottle\\train\\good\\208.png']

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 
                    'fryum', 'macaroni1', 'macaroni2', 
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

class VisADataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)