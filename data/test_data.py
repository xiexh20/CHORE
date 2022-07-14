"""
test dataset
"""
import numpy as np
from os.path import isfile
import pickle as pkl
import json
from psbody.mesh import Mesh
from lib_smpl.body_landmark import BodyLandmarks
from model.camera import KinectColorCamera
from .base_data import BaseDataset
import yaml, sys, cv2
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
SMPL_ASSETS_ROOT = paths["SMPL_ASSETS_ROOT"]


class TestData(BaseDataset):
    def __init__(self, data_paths, batch_size, num_workers,
                 dtype=np.float32,
                 image_size=(1024, 768),
                 input_type='RGBM3', crop_size=1200,
                 use_mean_center=False, **kwargs):
        super(TestData, self).__init__(data_paths, batch_size, num_workers, dtype=dtype)
        # input setting
        self.img_size = tuple(image_size)  # width, height
        self.input_type = input_type
        assert self.input_type == 'RGBM3'
        self.CROP_SIZE = np.array([crop_size, crop_size])  # crop from original high-reso image

        self.mean_crop_center = np.array([1008., 995.])  # computed from BEHAVE training set
        self.use_mean_center = use_mean_center # move cropped patch to the mean crop center

        self.depth = 2.2 if 'z_0' not in kwargs else kwargs.get('z_0')  # fixed smpl center depth
        self.landmark = BodyLandmarks(SMPL_ASSETS_ROOT)  # to get body keypoints from mocap mesh

        # for perspective projection
        self.camera = KinectColorCamera(crop_size)

    def get_item(self, idx):
        """
        load RGB and human+object masks, crop and scale the patch such that the person appears as if it is at z_0
        :param idx:
        :return:
        """
        rgb_file = self.data_paths[idx]
        images, center, resize_scale, scale, old_center = self.prepare_image_crop(rgb_file, False)
        res = {}
        res['images'] = images.astype(self.dtype)
        # res['image_file'] = rgb_file
        res['path'] = rgb_file
        res['resize_scale'] = resize_scale  # scale applied to original RGB image to make it similar to BEHAVE image size
        res['crop_scale'] = scale  # scale applied to the cropped patch
        res['crop_center'] = center.astype(self.dtype)
        res['old_crop_center'] = old_center # different from crop center if moved to mean center
        return res

    def prepare_image_crop(self, rgb_file, flip):
        assert not flip, 'for evaluation, do not do flip!'
        person_mask, obj_mask = self.load_masks(rgb_file, flip=False)

        bmin, bmax = self.masks2bbox([person_mask, obj_mask])
        width = bmax - bmin
        assert width[0] <= self.CROP_SIZE[0], 'crop width too small for image {} with bbox width {}'.format(rgb_file,
                                                                                                            width[0])
        assert width[1] <= self.CROP_SIZE[1], 'crop height too small for image {} with bbox height {}'.format(rgb_file,
                                                                                                              width[1])
        crop_center = (bmin + bmax) // 2
        rgb = self.load_rgb(rgb_file, flip=False)
        rh, rw = rgb.shape[:2]

        # resize to 1536x2048 pixel space
        if rw > rh:
            resize_scale = 2048 / rw
            newsize = (2048, int(rh * resize_scale))
        else:
            # resize along another direction
            resize_scale = 1536 / rh
            newsize = (int(rw * resize_scale), 1536)
        # resize everything to equivalent 2048 pixel space
        crop_center = np.round(resize_scale * crop_center)  # equivalent crop center in 2048px images
        rgb = cv2.resize(rgb, newsize)
        person_mask = cv2.resize(person_mask, newsize)
        obj_mask = cv2.resize(obj_mask, newsize)

        # everything should be the same as in training time!
        kpts = self.load_j2d(rgb_file)
        if np.sum(kpts[:, 2]) == 0:
            raise ValueError('no valid person keypoints in image {}'.format(rgb_file))
        scaled_kpts = kpts.copy()
        scaled_kpts[:, :2] *= resize_scale  # must be in 2048p image

        scale = self.fullbody_crop(scaled_kpts, rgb_file)
        # obtain the new cropping square
        # scale > 1.0: the person appears too large in current patch, make a larger crop to make it smaller
        crop_size = scale * self.CROP_SIZE

        # change crop center
        rgb = self.pad_image(rgb, crop_center)
        person_mask = self.pad_image(person_mask, crop_center)
        obj_mask = self.pad_image(obj_mask, crop_center)
        old_center = crop_center.copy()
        crop_center = self.change_crop_center(crop_center)
        # print("{} old center: {}, new center: {}".format(rgb_file, old_center, crop_center))

        rgb = self.resize(self.crop(rgb, crop_center, crop_size), self.img_size) / 255.
        person_mask = self.resize(self.crop(person_mask, crop_center, crop_size), self.img_size) / 255.
        obj_mask = self.resize(self.crop(obj_mask, crop_center, crop_size), self.img_size) / 255.

        images = self.compose_images(obj_mask, person_mask, rgb)

        # save cropping information for rendering
        outfile = rgb_file.replace(f".color.jpg", '.crop_info.pkl')
        if not isfile(outfile):
            crop_info = {
                'rgb_newsize': np.array([2048, 1536]),
                "resize_scale": resize_scale,
                "crop_center": old_center,
                "crop_scale": scale,
                "crop_size": crop_size
            }
            pkl.dump(crop_info, open(outfile, 'wb'))

        return images.transpose((2, 0, 1)).astype(self.dtype), crop_center, resize_scale, scale, old_center

    def change_crop_center(self, crop_center):
        if self.use_mean_center:
            return self.mean_crop_center.copy()
        else:
            return crop_center

    def pad_image(self, img, crop_center):
        if not self.use_mean_center:
            return img
        # pad image such that the cropping center becomes mean center
        h, w = img.shape[:2]
        top_left = (self.mean_crop_center - crop_center).astype(int)
        bottom_right = (np.array([w, h]) + top_left)

        kw, kh = 2048, 1536  # kinect image size
        new_size = np.maximum(np.array([kw, kh]), bottom_right).astype(
            int)  # have a large enough image to make sure no body part is cut
        if img.ndim == 3:
            new_img = np.zeros((new_size[1], new_size[0], 3))
        else:
            new_img = np.zeros((new_size[1], new_size[0]))

        # find the indexing in new image
        x1y1 = np.maximum(np.zeros(2), top_left).astype(int)
        x2y2 = np.minimum(np.array([kw, kh]), bottom_right).astype(int)

        # find the indexing in input image
        x1 = max(0, -top_left[0])
        y1 = max(0, -top_left[1])
        x2 = min(w, w - (bottom_right[0] - kw))
        y2 = min(h, h - (bottom_right[1] - kh))

        new_img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]] = img[y1:y2, x1:x2]

        return new_img

    def load_j2d(self, rgb_file):
        """
        load 2D body keypoints, in original image coordinate (before any crop or scale)
        :param rgb_file:
        :return:
        """
        json_path = rgb_file.replace('.color.jpg', '.color.json')
        data = json.load(open(json_path))
        J2d = np.array(data["body_joints"]).reshape((-1, 3))
        return J2d

    def fullbody_crop(self, pts, rgb_file=None):
        """
        compute the crop and scale parameters such that after crop and scale, the person
        looks as if it is at z_0 under a certain camera projection
        :param pts: (N, 3), 2D body keypoints detected by openpose, the third column is confidence
        :param rgb_file: input RGB image file path
        :return: a factor to scale the cropped patch
        """
        if np.sum(pts[:, 2]) == 0:
            print("Warning, using mean radius!")
            return None, 1.0
        mesh = self.load_mocap_mesh(rgb_file)
        # move to 2.2m depth
        mesh.v = mesh.v - np.mean(mesh.v, 0) + np.array([0, 0, self.depth])
        j3d = self.landmark.get_body_kpts(mesh)
        # project to image plane
        j3d_proj = self.persp_proj(j3d)
        valid_mask = pts[:, 2] > 0.3

        j2d = pts[valid_mask]
        j2d_mocap = j3d_proj[valid_mask]

        # get the bbox of joints
        _, width = self.get_bbox(j2d[:, :2])
        _, width_mocap = self.get_bbox(j2d_mocap[:, :2]) # this is the target width

        # compute scaling factor
        w, h = width[0], width[1]
        wm, hm = width_mocap[0], width_mocap[1]
        if w >= h and wm >= hm:
            # use width to compute scale
            scale = w/wm # if the actual j2d is smaller than target j2d ==>> person far away, we want a smaller crop,
            # so after resizing, the person looks bigger (virtually moved closer)
        else:
            # use height to compute scale
            scale = h/hm
        return scale

    def load_mocap_mesh(self, rgb_file):
        mesh_file = rgb_file.replace(f'.color.jpg', '.mocap.ply')
        m = Mesh()
        m.load_from_file(mesh_file)
        return m

    def persp_proj(self, points):
        "perspective projection"
        px, py = self.camera.project_screen(points) # project back to original kinect image size
        confidence = np.ones_like(px) # confidence is always 1
        return np.concatenate([px, py, confidence], 1)

    def get_bbox(self, j2d, exp=1.1):
        "get the bounding box of all 2d joints"
        bmin = np.min(j2d, 0)
        bmax = np.max(j2d, 0)
        return bmin, (bmax - bmin)*exp