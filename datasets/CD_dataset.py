"""
变化检测数据集
"""
import torchvision.transforms.functional as TF
import os
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils import data

from datasets.data_utils import CDDataAugmentation
from sklearn.model_selection import train_test_split
from tqdm import tqdm
"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "images"
IMG_POST_FOLDER_NAME = 'images'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "targets"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir, split, img_name):
    # img_name = img_name + "_post_disaster.png"
    return os.path.join(root_dir, split, 'B', img_name)


def get_img_path(root_dir, split, img_name):
    # img_name = img_name + "_pre_disaster.png"
    return os.path.join(root_dir, split, 'A', img_name)


def get_label_path(root_dir, split, img_name):
    #img_name = img_name + "_post_disaster.png"
    # img_name = img_name.replace("images", "masks") + "_post_disaster.png"
    return os.path.join(root_dir, split, 'label', img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=1024, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = os.listdir(os.path.join(self.root_dir, split, 'A'))

        # self.img_name_list = load_img_name_list(self.list_path)
        # tmp_lst = [("hurricane" in x) for x in self.img_name_list]
        # self.img_name_list = np.array(self.img_name_list)[tmp_lst].tolist()
        
        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                with_random_resize=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, patch=None):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        self.split = split
        self.patch = patch

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        
        L_path = get_label_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        label = np.array(Image.open(L_path), dtype=np.uint8)
        

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
           label = label // 255

        # # 2 class model
        # label[label <= 1] = 0
        # label[label > 1] = 1
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor, patch=self.patch)
        # [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor, split=self.split)
        return {'name': name, 'A': img, 'B': img_B, 'L': label}


class xBDataset(data.Dataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(xBDataset, self).__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.label_transform = label_transform
        self.split = split

        train_dirs = ['data/xbd/train'] # fix path!!
        all_files = []
        for d in train_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
#                if ('_pre_disaster.png' in f) and (('hurricane-harvey' in f) | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f) | ('palu-tsunami' in     f)):
                 if ('_pre_disaster.png' in f):
                    all_files.append(os.path.join(d, 'images', f))

        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=10)
        if split == 'train':
            self.img_name_list = np.array(all_files)[train_idxs]
        elif split == 'val':
            self.img_name_list = np.array(all_files)[val_idxs]
        elif split == 'test':
            self.img_name_list = np.array(all_files)[val_idxs]

    def __getitem__(self, index):
        fn = self.img_name_list[index]
        
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)
        label = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        label[label <= 2] = 0
        label[label > 2] = 1
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        name = fn.split('/')[-1]
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)


class xBDatasetMulti(data.Dataset):
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(xBDatasetMulti, self).__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.label_transform = label_transform
        self.split = split

        #train_dirs = [f'{self.root_dir}/train', f'{self.root_dir}/tier3', f'{self.root_dir}/test', f'{self.root_dir}/hold'] # fix path!!
        #train_dirs = [f'{self.root_dir}/train', f'{self.root_dir}/hold',f'{self.root_dir}/tier3']
        #train_dirs = [f'{self.root_dir}/train']
        #train_dirs = [f'{self.root_dir}/hold']
        #train_dirs = [f'{self.root_dir}/tier3']
        train_dirs = [f'{self.root_dir}/train']
        all_files = []
        for d in train_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                # if ('_pre_disaster.png' in f) and (('hurricane-harvey' in f) | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f) | ('palu-tsunami' in     f)):
                if '_pre_disaster.png' in f:
                    all_files.append(os.path.join(d, 'images', f))

        # Upsampling
        file_classes = []
        for fn in all_files:
            fl = np.zeros((4,), dtype=bool)
            #print(fn)
            path = fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')
            #print(path)
            msk1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            #print('This is the shape of the mask',msk1.shape)
            #print(msk1.shape)
            for c in range(1, 5):
                fl[c-1] = c in msk1
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)
        for i in range(len(file_classes)):
            im = all_files[i]
            if file_classes[i, 1:].max():
                all_files.append(im)
            if file_classes[i, 1:3].max():
                all_files.append(im)

        # train test split
        print(f'Number of Training Images {len(all_files)}')
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=10)
        if split == 'train':
            self.img_name_list = np.array(all_files)[train_idxs]
        elif split == 'val':
            self.img_name_list = np.array(all_files)[val_idxs]
        elif split == 'test':
            self.img_name_list = np.array(all_files)[val_idxs]


    def __getitem__(self, index):
        #print(len(self.img_name_list))
        fn = self.img_name_list[index]
        
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)
        path = fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')
        #print(path)
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if self.split == 'train':
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
            #print(label.shape)
        else:
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        name = fn.split('/')[-1]
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)
        
        
        


class CC(data.Dataset):
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CC, self).__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val

        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.label_transform = label_transform
        self.split = split

        # train_dirs = [f'{self.root_dir}/train', f'{self.root_dir}/tier3', f'{self.root_dir}/test', f'{self.root_dir}/hold'] # fix path!!
        # train_dirs = [f'{self.root_dir}/train', f'{self.root_dir}/hold',f'{self.root_dir}/tier3']
        # train_dirs = [f'{self.root_dir}/train']
        # train_dirs = [f'{self.root_dir}/hold']
        # train_dirs = [f'{self.root_dir}/tier3']
        train_dirs = [f'{self.root_dir}/all_new_data']
        all_files = []
        for d in train_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                # if ('_pre_disaster.png' in f) and (('hurricane-harvey' in f) | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f) | ('palu-tsunami' in     f)):
                if '_pre.png' in f:
                    all_files.append(os.path.join(d, 'images', f))

        #Upsampling
        file_classes = []
        for fn in all_files:
            fl = np.zeros((4,), dtype=bool)
            # print(fn)
            path = fn.replace('/images/', '/masks/').replace('_pre', '_post')
            # print(path)
            msk1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            # print('This is the shape of the mask',msk1.shape)
            # print(msk1.shape)
            for c in range(1, 5):
                fl[c - 1] = c in msk1
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)
        for i in range(len(file_classes)):
            im = all_files[i]
            if file_classes[i, 1:].max():
                all_files.append(im)
            if file_classes[i, 1:3].max():
                all_files.append(im)

        # train test split
        print(f'Number of Training Images {len(all_files)}')
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=10)
        if split == 'train':
            self.img_name_list = np.array(all_files)[train_idxs]
        elif split == 'val':
            self.img_name_list = np.array(all_files)[val_idxs]
        elif split == 'test':
            self.img_name_list = np.array(all_files)[val_idxs]

    def __getitem__(self, index):
        # print(len(self.img_name_list))
        fn = self.img_name_list[index]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre', '_post'), cv2.IMREAD_COLOR)
        path = fn.replace('/images/', '/masks/').replace('_pre', '_post')
        # print(path)
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if self.split == 'train':
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
            # print(label.shape)
        else:
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        name = fn.split('/')[-1]
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)

    def __getitem__(self, index):
        # print(len(self.img_name_list))
        fn = self.img_name_list[index]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre', '_post'), cv2.IMREAD_COLOR)
        path = fn.replace('/images/', '/masks/').replace('_pre', '_post')
        # print(path)
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(f'labels shape', label.shape)

        if self.split == 'train':
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
            # print(label.shape)
        else:
            [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        # name = fn.split('/')[-1]
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)




class TestInput(data.Dataset):
    def __init__(self, root_dir, img_size, split='train', is_train=False, label_transform=None,
                 to_tensor=True):
        super(TestInput, self).__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val

        self.to_tensor = to_tensor
        self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.label_transform = label_transform
        self.split = split

        train_dirs = [f'{self.root_dir}/test_cases']
        all_files = []
        for d in tqdm(train_dirs, disable=True ,desc='The System is Loading Patches'):
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                if '_pre.png' in f:
                    all_files.append(os.path.join(d, 'images', f))
        print(f'Number of Testing Images {len(all_files)}')
        self.img_name_list = np.array(all_files)

    def __getitem__(self, index):
        fn = self.img_name_list[index]
        name = fn.split('/')[-1].split('.')[0].replace('_pre', '_post')
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre', '_post'), cv2.IMREAD_COLOR)
        imgs = [img, img_B]
        if self.to_tensor:
            imgs = [TF.to_tensor(img) for img in imgs]
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    for img in imgs]

        img, img_B = imgs
        return {'name': fn, 'A': img, 'B': img_B, 'n': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)
