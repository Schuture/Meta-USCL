import os
import re
import random
import pickle

from PIL import Image
import jpeg4py as jpeg # use jpeg4py package can shorten the epoch time to 85%
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

np.random.seed(0)


def load_image(path_img, aug_first):
    try:
        if aug_first: # get PIL image
            return Image.fromarray(jpeg.JPEG(path_img).decode()).convert('RGB')
        else: # get ndarray
            return jpeg.JPEG(path_img).decode()
    except:
        if aug_first:
            return Image.open(path_img).convert('RGB')
        else:
            return np.array(Image.open(path_img).convert('RGB'))


class COVIDDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        with open(data_dir, 'rb') as f: # 导入所有数据
            X_train, y_train, X_test, y_test = pickle.load(f)
        if train:
            self.X, self.y = X_train, y_train # [N,C,H,W], [N]
        else:
            self.X, self.y = X_test, y_test # [N,C,H,W], [N]
        self.transform = transform
    
    def __getitem__(self, index):
        img_arr = self.X[index].transpose(1,2,0)
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB') # 0~255
        # label = self.y[index]

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return img1, img2 # contrastive learning

    def __len__(self):
        return len(self.y)


class USDataset_image(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img = self.data_info[index]
        img1 = Image.open(path_img).convert('RGB')
        img2 = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for dataset in ['Butterfly_Dataset', 'CLUST_50', 'COVID19_LUSMS', 'Liver_Fibrosis']:
            dataset_dir = os.path.join(data_dir, dataset)
            for root, dirs, _ in os.walk(dataset_dir):
                for sub_dir in dirs:
                    img_names = os.listdir(os.path.join(root, sub_dir))
                    # all image names in a video
                    img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))
                    if not img_names: # null video
                        print('Find error in', os.path.join(root, sub_dir))
                        continue
                        
                    for i in range(len(img_names)):
                        img_name = img_names[i]
                        path_img = os.path.join(root, sub_dir, img_name)
                        data_info.append(path_img)

        return data_info


class USDataset_video(Dataset):

    def __init__(self, data_dir, mixup='standard', transform=None,
                 max_dist=16, samples=3, aug_first=False):
        '''
        Ultrasound self-supervised training Dataset
        Augment multiple different images of a video.
        Positive pair interpolator is implemented in __getitem__().
        
        :param data_dir: str, data directory
        :param transform: torch.transform，data augmentation
        '''
        self.data_info = self.get_img_info(data_dir)
        self.mixup = mixup
        if mixup and samples >= 3:
            print('Augmented from 3 frames mixup')
        elif samples >= 2:
            print('Augmented from 2 random frames')
        else:
            print('Augmented from 1 random frame')
        self.transform = transform
        self.max_dist = max_dist
        self.samples = samples
        self.aug_first = aug_first

    def __getitem__(self, index):
        path_imgs = self.data_info[index]
        img = []
        # If there are 3 or more images for this video, we use mixup to generate positives.
        if self.mixup and len(path_imgs) >= 3 and self.samples >= 3:
            # Pick n images under the distance restriction, images are already sorted.
            n = min(len(path_imgs), self.samples)
            indices = [0 for i in range(n)]
            # the distance of other n-1 samples to the first one is in [1, max_dist]
            distances = sorted(random.sample(list(range(1, min(len(path_imgs) - 1, self.max_dist) + 1)), n-1))
            indices[0] = random.choice(list(range(len(path_imgs) - distances[n-2]))) # index for image 1
            for i in range(1, n):
                indices[i] = indices[0] + distances[i-1]
            for i in range(n):
                img.append(load_image(path_imgs[indices[i]], self.aug_first))
            # standard mixup for old SPG, n=3
            if self.mixup == 'standard':
                alpha, beta = np.random.beta(3, 5), np.random.beta(3, 5)
            # inner point of a convex hull
            if self.mixup == 'inner': # weight generation scheme, larger possibility for extreme points
                alpha = [np.random.rand() for i in range(n-1)]
                alpha.sort()
                alpha = [0] + alpha + [1]
                alpha = [alpha[i+1] - alpha[i] for i in range(n)]
                beta = [np.random.rand() for i in range(n-1)]
                beta.sort()
                beta = [0] + beta + [1]
                beta = [beta[i+1] - beta[i] for i in range(n)]
        # This video only has 2 images, or we only pick 2 frames from a video.
        elif len(path_imgs) >= 2 and self.samples >= 2:
            # Pick two frames under the restriction of max distance.
            distance = random.choice(list(range(1, min(len(path_imgs) - 1, self.max_dist) + 1))) # distance of 2 samples
            idx1 = random.choice(list(range(len(path_imgs) - distance))) # index of the first sample
            idx2 = idx1 + distance
            img = [load_image(path_imgs[idx1], True),
                   load_image(path_imgs[idx2], True)]
        # Use 1 frame from a video to get 2 samples.
        else:
            img = [load_image(path_imgs[0], True),
                   load_image(path_imgs[0], True)]

        if self.transform is not None:
            if len(img) == 2:
                img1 = self.transform(img[0])
                img2 = self.transform(img[1])
            elif self.mixup == 'standard':  # 3 images => 2 mixed images
                if self.aug_first:
                    img1 = self.transform(img[0])
                    img2 = self.transform(img[1])
                    img3 = self.transform(img[2])
                    img1, img2 = img1 * (1-alpha) + img2 * alpha, img2 * beta + img3 * (1-beta)
                else:
                    img1, img2 = img[0] * (1-alpha) + img[1] * alpha, img[1] * beta + img[2] * (1-beta)
                    img1, img2 = Image.fromarray(np.uint8(img1)), Image.fromarray(np.uint8(img2))
                    img1, img2 = self.transform(img1), self.transform(img2)
            elif self.mixup == 'inner':  # n images => 2 mixed images
                if self.aug_first:
                    transformed_img = self.transform(img[0])
                    img1, img2 = alpha[0] * transformed_img, beta[0] * transformed_img
                    for i in range(1, n):
                        transformed_img = self.transform(img[i])
                        img1 += alpha[i] * transformed_img
                        img2 += beta[i] * transformed_img
                else:
                    img1, img2 = alpha[0] * img[0], beta[0] * img[0]
                    for i in range(1, n):
                        img1 += alpha[i] * img[i]
                        img2 += beta[i] * img[i]
                    img1, img2 = Image.fromarray(np.uint8(img1)), Image.fromarray(np.uint8(img2))
                    img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:  # traverse video dirs
                # Get all image names in the video dir.
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))
                
                # Sort the image names according to the frame indices.
                img_name_with_idx = []
                for img_name in img_names:
                    m1 = re.search('frame', img_name)
                    m2 = re.search('.jpg', img_name)
                    if not m2:
                        m2 = re.search('.png', img_name)
                    idx = int(img_name[m1.span(0)[1]: m2.span(0)[0]])
                    img_name_with_idx.append((idx, img_name))
                img_name_with_idx.sort()

                # Put all image paths from this video to a list.
                path_imgs = []
                for i in range(len(img_name_with_idx)):
                    img_name = img_name_with_idx[i][1]
                    path_img = os.path.join(root, sub_dir, img_name)
                    path_imgs.append(path_img)
                    
                data_info.append(path_imgs)

        random.shuffle(data_info)

        return data_info#[:300]


def get_transforms(input_shape, cropping_size=0.7, color_jitter=0.8):
    ''' 
    Get a set of data augmentation transformations for training and testing.
    Random Crop (resize to original size) + Random color distortion + Gaussian Blur
    ''' 
    s1 = cropping_size
    s2 = color_jitter
    data_transforms = transforms.Compose([transforms.Resize((input_shape, input_shape)),
                                          transforms.RandomResizedCrop(size=input_shape, scale=(s1, 1.0), ratio=(0.8, 1.25)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(0.8 * s2, 0.8 * s2, 0.8 * s2, 0.2 * s2),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.18,0.18,0.18],
                                                               std=[0.24,0.24,0.24])
                                          ])

    return data_transforms


def get_dataloaders(dataset='US-4', data_path=None, meta=False, downstream=False,
                    batch_size=32, valid_batch_size=32, meta_batch_size=32,
                    workers=0, valid_ratio=0.2, mixup='standard',
                    max_dist=16, samples=3, input_shape=32, aug_first=False,
                    cropping_size=0.7, color_jitter=0.8):
    ''' Get dataloaders for main training, meta training, and validation. '''
    
    data_augment = get_transforms(input_shape, cropping_size, color_jitter) # 56x56 images or smaller will exceptionally speed up the training
    print('Data augmentation:')
    print(data_augment)

    if dataset == 'US-4':
        print('\ntraining dataset:')
        train_dataset = USDataset_video(data_path, mixup,
                                        transform=data_augment,
                                        max_dist=max_dist, samples=samples,
                                        aug_first=aug_first)
        if meta:
            print('\nmeta dataset:')
            meta_dataset = USDataset_video(data_path, mixup,
                                           transform=data_augment,
                                           max_dist=max_dist, samples=1,
                                           aug_first=aug_first)
    else:
        raise ValueError('Only US-4 dataset is supported for now.')
    
    # Obtain indices for samples that will be used for training / validation.
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_ratio * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    print('\nTraining data: {}, validaiton data: {}'.format(num_train-split, split))

    # Define samplers for obtaining training and validation batches.
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Data loaders for training, validation, and meta learning.
    # 'drop_last' should be False to avoid data shortage and avoid the StopIteration error.
    # 'sampler' option is mutually exclusive with shuffle.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=workers, drop_last=False, shuffle=False)
    valid_loader = DataLoader(train_dataset, batch_size=valid_batch_size, sampler=valid_sampler,
                              num_workers=workers, drop_last=False, shuffle=False)
    
    meta_loader = None if not meta else DataLoader(meta_dataset, batch_size=meta_batch_size, sampler=valid_sampler,
                                                   num_workers=workers, drop_last=False)
                              
    return train_loader, valid_loader, meta_loader



