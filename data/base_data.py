"""
basic operations for the dataset
"""
import os
import numpy as np
import cv2
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torchvision.transforms as transforms
from PIL.ImageFilter import GaussianBlur


class BaseDataset(Dataset):
    def __init__(self, data_paths, batch_size, num_workers, dtype=np.float32,
                 aug_blur=0.0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_paths = data_paths
        self.dtype = dtype

        # data augmentation
        self.aug_blur = aug_blur

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # ret = self.get_item(idx)
        # return ret
        try:
            ret = self.get_item(idx)
            return ret
        except Exception as e:
            print(e)
            ridx = np.random.randint(0, len(self.data_paths))
            print(f"failed on {self.data_paths[idx]}, retrying {self.data_paths[ridx]}")
            return self[ridx]

    def get_item(self, idx):
        raise NotImplemented

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def get_loader(self, shuffle=True, rank=-1, world_size=-1):
        if world_size>0:
            "loader for multiple gpu training"
            sampler = DistributedSampler(
                    self,
                    num_replicas=world_size,
                    rank=rank)
            loader = DataLoader(dataset=self, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers,
                                sampler=sampler,
                                pin_memory=True,
                                drop_last=True)
            return loader

        else:
            return DataLoader(
                self, batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                worker_init_fn=self.worker_init_fn,
                drop_last=False)

    def load_masks(self, rgb_file, flip=False):
        person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.jpg")
        if not osp.isfile(person_mask_file):
            person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.png")
        obj_mask_file = rgb_file.replace('.color.jpg', ".obj_rend_mask.jpg")
        if not osp.isfile(obj_mask_file):
            obj_mask_file = rgb_file.replace('.color.jpg', ".obj_mask.jpg")
            if not osp.isfile(obj_mask_file):
                obj_mask_file = rgb_file.replace('.color.jpg', ".obj_mask.png")
        person_mask = cv2.imread(person_mask_file, cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(obj_mask_file, cv2.IMREAD_GRAYSCALE)

        if flip:
            person_mask = self.flip_image(person_mask)
            obj_mask = self.flip_image(obj_mask)

        return person_mask, obj_mask

    def flip_image(self, img):
        img = Image.fromarray(img)
        flipped = transforms.RandomHorizontalFlip(p=1.0)(img)
        img = np.array(flipped)
        return img

    def masks2bbox(self, masks, thres=127):
        """
        convert a list of masks to an bbox of format xyxy
        :param masks:
        :param thres:
        :return:
        """
        mask_comb = np.zeros_like(masks[0])
        for m in masks:
            mask_comb += m
        mask_comb = np.clip(mask_comb, 0, 255)
        ret, threshed_img = cv2.threshold(mask_comb, thres, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bmin, bmax = np.array([50000, 50000]), np.array([-100, -100])
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            bmin = np.minimum(bmin, np.array([x, y]))
            bmax = np.maximum(bmax, np.array([x+w, y+h]))
        return bmin, bmax

    def load_rgb(self, rgb_file, flip=False):
        rgb = np.array(Image.open(rgb_file))
        if flip:
            rgb = self.flip_image(rgb)
        rgb = self.blur_image(rgb)
        return rgb

    def blur_image(self, img):
        assert isinstance(img, np.ndarray)
        if self.aug_blur > 0.000001:
            x = np.random.uniform(0, self.aug_blur)*255. # input image is in range [0, 255]
            blur = GaussianBlur(x)
            img = Image.fromarray(img)
            return np.array(img.filter(blur))
        return img

    def crop(self, img, center, crop_size):
        """
        crop image around the given center, pad zeros for boraders
        :param img:
        :param center:
        :param crop_size: size of the resulting crop
        :return:
        """
        assert isinstance(img, np.ndarray)
        h, w = img.shape[:2]
        topleft = np.round(center - crop_size / 2).astype(int)
        bottom_right = np.round(center + crop_size / 2).astype(int)

        x1 = max(0, topleft[0])
        y1 = max(0, topleft[1])
        x2 = min(w - 1, bottom_right[0])
        y2 = min(h - 1, bottom_right[1])
        cropped = img[y1:y2, x1:x2]

        p1 = max(0, -topleft[0])  # padding in x, top
        p2 = max(0, -topleft[1])  # padding in y, top
        p3 = max(0, bottom_right[0] - w+1)  # padding in x, bottom
        p4 = max(0, bottom_right[1] - h+1)  # padding in y, bottom

        dim = len(img.shape)
        if dim == 3:
            padded = np.pad(cropped, [[p2, p4], [p1, p3], [0, 0]])
        elif dim == 2:
            padded = np.pad(cropped, [[p2, p4], [p1, p3]])
        else:
            raise NotImplemented
        return padded

    def resize(self, img, img_size, mode=cv2.INTER_LINEAR):
        """
        resize image to the input
        :param img:
        :param img_size: (width, height) of the target image size
        :param mode:
        :return:
        """
        h, w = img.shape[:2]
        load_ratio = 1.0 * w / h
        netin_ratio = 1.0 * img_size[0] / img_size[1]
        assert load_ratio == netin_ratio, "image aspect ration not matching, given image: {}, net input: {}".format(img.shape, self.img_size)
        resized = cv2.resize(img, img_size, interpolation=mode)
        return resized

    def compose_images(self, obj_mask, person_mask, rgb):
        """
        mask background out, and stack RGB, h+o masks
        :param obj_mask:
        :param person_mask:
        :param rgb:
        :return:
        """
        # assert self.input_type == 'RGBM3'
        # mask background out
        mask_comb = (person_mask > 0.5) | (obj_mask > 0.5)
        rgb = rgb * np.expand_dims(mask_comb, -1)
        images = np.dstack((rgb, person_mask, obj_mask))
        return images
