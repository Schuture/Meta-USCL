import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
 

class USSegDataset(torch.utils.data.Dataset):
    '''
    Dataset for Ultrasound segmentation. File structure：
    UDIAT_Dataset_B:
        GT: masks for each image，png file，bg: 0, lesion: 255
        original: data images，png
    '''
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "original"))))
        self.imgs = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), self.imgs))
        self.masks = list(sorted(os.listdir(os.path.join(root, "GT"))))
        self.masks = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), self.masks))
 
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "original", self.imgs[idx])
        mask_path = os.path.join(self.root, "GT", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)
 
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # boundary of masks as bbox
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)














