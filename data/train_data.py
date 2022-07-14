"""
dataloader to train the model
"""
import numpy as np

from .base_data import BaseDataset


class BehaveDataset(BaseDataset):
    def __init__(self, data_paths, batch_size, phase,
                 num_workers,
                 dtype=np.float32,
                 total_samplenum=20000,
                 image_size=(1024, 768),
                 ratios=[0.01, 0.49, 0.5],
                 sigmas=[0.08, 0.02, 0.003],
                 input_type='RGBM3',
                 random_flip=False,
                 aug_blur=0.,
                 crop_size=1200,
                 **kwargs
                 ):
        super(BehaveDataset, self).__init__(data_paths, batch_size, num_workers, dtype, aug_blur)
        self.phase = phase
        assert phase in ['train', 'val', 'test']

        # input setting
        self.img_size = tuple(image_size)  # width, height
        self.input_type = input_type
        self.CROP_SIZE = np.array([crop_size, crop_size]) # crop from original high-reso image

        # surface sampling settings
        self.total_sample_num = total_samplenum
        self.sample_nums = [int(total_samplenum * r) for r in ratios]
        self.sigmas = sigmas

        # data augmentation
        self.random_flip = random_flip
        self.aug_blur = aug_blur

        # kinect camera parameters
        self.center = np.array([1018.952, 779.486])
        self.focal = np.array([979.7844, 979.840])
        self.depth = 2.2 if 'z_0' not in kwargs else kwargs.get('z_0')  # fixed smpl center depth

    def get_item(self, idx):
        path = self.data_paths[idx]
        if self.phase == 'train':
            flip = (np.random.rand() > 0.5) & self.random_flip
        else:
            flip = False

        if flip:
            path = path.replace(".npz", "_flip.npz") # different GT for part labels

        data = np.load(path, allow_pickle=True)
        res = self.get_samples(data)
        images, center = self.prepare_image_crop(data, flip)

        # add additional info data
        res['path'] = path
        res['kid'] = 1
        res['images'] = images.astype(self.dtype)
        res['image_file'] = str(data['image_file'])
        res['flip'] = flip
        res['crop_center'] = center.astype(self.dtype)
        return res

    def get_samples(self, data):
        """
        sample points from boundary samples to train the model
        :param data:
        :return:
        """
        dfs_h, dfs_o = [], []
        part_labels = []
        points_all = []

        for sigma, sample_num in zip(self.sigmas, self.sample_nums):
            sample_key = 'sigma{}'.format(sigma)
            points = data['points'].item()[sample_key]
            choice = np.random.choice(points.shape[0], sample_num, replace=False)

            points_all.append(points[choice])
            dfs_h.append(data['dist_h'].item()[sample_key][choice])
            dfs_o.append(data['dist_o'].item()[sample_key][choice])
            part_labels.append(data['parts'].item()[sample_key][choice])

        points_all = np.concatenate(points_all, 0).astype(np.float32)
        dfs_h = np.concatenate(dfs_h, 0)
        dfs_o = np.concatenate(dfs_o, 0)
        parts = np.concatenate(part_labels, 0)

        pca_axis = data['pca_axis']
        pca_axis = np.repeat(pca_axis[None], points_all.shape[0], axis=0).transpose(1, 2, 0) # (3, 3, N)

        data_dict = {
            'points':points_all,
            'df_h':dfs_h.astype(np.float32),
            'df_o':dfs_o.astype(np.float32),
            'labels':parts.astype(np.float32),
            'pca_axis':pca_axis.astype(np.float32)
        }

        offset = data['smpl_center']
        assert np.abs(offset[2]-2.2)<1e-6, 'invalid offset value found: {}'.format(offset)
        data_dict['body_center'] = offset.astype(self.dtype)

        # object center is relative to SMPL center
        obj = data['obj_center'].astype(np.float32) - data_dict['body_center'].astype(np.float32)
        data_dict['obj_center'] = np.repeat(obj[None], data_dict['points'].shape[0], axis=0).transpose(1, 0)  # (3, N)

        return data_dict

    def get_crop_center(self, rgb_file):
        """
        load RGB image and human+object masks, find the crop center as the center of h+o bboxes
        :param rgb_file:
        :param flip:
        :return:
        """
        person_mask, obj_mask = self.load_masks(rgb_file)

        bmin, bmax = self.masks2bbox([person_mask, obj_mask]) # crop using full bbox
        crop_center = (bmin + bmax) // 2
        assert np.sum(crop_center > 0) == 2, 'invalid bbox found'
        ih, iw = person_mask.shape[:2]
        assert crop_center[0] < iw and crop_center[0] >0, 'invalid crop center value {} for image {}'.format(crop_center, rgb_file)
        assert crop_center[1] < iw and crop_center[1] > 0, 'invalid crop center value {} for image {}'.format(
            crop_center, rgb_file)

        return crop_center

    def prepare_image_crop(self, data, flip):
        "crop around the center"
        rgb_file = str(data['image_file'])
        person_mask, obj_mask = self.load_masks(rgb_file, flip)
        crop_center = self.get_crop_center(rgb_file)
        rgb = self.load_rgb(rgb_file, flip)

        # do crop
        rgb = self.resize(self.crop(rgb, crop_center, self.CROP_SIZE), self.img_size) / 255.
        person_mask = self.resize(self.crop(person_mask, crop_center, self.CROP_SIZE), self.img_size) / 255.
        obj_mask = self.resize(self.crop(obj_mask, crop_center, self.CROP_SIZE), self.img_size) / 255.

        # images = self.compose_images(crop_center, flip, obj_mask, person_mask, rgb, rgb_file)
        images = self.compose_images(obj_mask, person_mask, rgb)

        return images.transpose((2, 0, 1)).astype(self.dtype), crop_center

