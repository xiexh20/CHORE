"""
Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import cv2
import numpy as np
import copy
import torch
from psbody.mesh import Mesh
from os.path import isfile
import neural_renderer as nr
from neural_renderer.renderer import Renderer

SMPL_OBJ_COLOR_LIST = [
        [0.65098039, 0.74117647, 0.85882353],  # SMPL
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
    ]


class NrWrapper:
    "simple wrapper for neural renderer"
    def __init__(self, device='cuda:0', image_size=1024, colors=None):
        self.device = device
        if colors is None:
            self.colors = copy.deepcopy(SMPL_OBJ_COLOR_LIST)
        else:
            self.colors = colors
        self.smpl_color = SMPL_OBJ_COLOR_LIST[0]
        self.obj_color = SMPL_OBJ_COLOR_LIST[1]
        self.front_renderer = setup_renderer(image_size=image_size)

    def render(self, renderer, verts, faces, texts):
        "return image in range [0, 1]"
        image, depth, mask = renderer.render(vertices=verts, faces=faces,
                                             textures=texts)  # the second return value is depth
        rend = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)[:, :, :3]
        mask = mask[0].detach().cpu().numpy().astype(bool)
        return rend, mask

    def render_meshes(self, renderer, meshes:list, colors=None):
        """
        render smpl and object mesh
        :param renderer: neural_renderer renderer
        :param meshes: list of SMPL and object mesh, psbody.mesh.Mesh
        :param colors: color for SMPL and object faces
        :return:
        """
        verts, faces, texts = self.prepare_render(meshes, colors)
        return self.render(renderer, verts, faces, texts)

    def prepare_render(self, meshes, colors=None):
        faces_list = []
        verts_list = []
        color_list = []
        render_color = self.colors if colors is None else colors
        for m, c in zip(meshes, render_color):
            faces_list.append(torch.tensor(m.f.astype(np.int32), dtype=torch.int32).to(self.device))
            verts_list.append(torch.tensor(m.v, dtype=torch.float32).to(self.device).unsqueeze(0))
            color_list.append(c)

        verts_comb = torch.cat(verts_list, 1)
        faces, textures = get_faces_and_textures(verts_list, faces_list, colors_list=color_list)

        return verts_comb, faces, textures

    def prepare_side_rend(self, meshes, maxd=1.5, colors=None):
        meshes = self.rotate_meshes(meshes) # neural renderer look_at mode and normal mode have different camera coordinate convertion!
        meshes_norm, scale = self.normalize_meshes(meshes, maxd=maxd, ret_scale=True)
        verts, faces, texts = self.prepare_render(meshes_norm, colors=colors)
        # center and mirror
        center = torch.mean(verts, 1)
        verts = verts - center

        return faces, texts, verts

    @staticmethod
    def normalize_meshes(meshes, maxd=2.0, ret_scale=False):
        "normalize the meshes, the larger maxd, the larger rendered mesh "
        scale = cal_norm_scale(meshes, maxd)
        for m in meshes:
            m.v = m.v * scale
        if ret_scale:
            return meshes, scale
        return meshes

    def rotate_meshes(self, meshes):
        rot = np.eye(3)
        rot[1, 1] = -1
        meshes_ret = []
        for m in meshes:
            mc = self.copy_mesh(m)
            mc.v = np.matmul(mc.v, rot.T)
            meshes_ret.append(mc)
        return meshes_ret

    def copy_mesh(self, mesh: Mesh):
        m = Mesh(v=mesh.v)
        if hasattr(mesh, 'f'):
            m.f = mesh.f.copy()
        if hasattr(mesh, 'vc'):
            m.vc = np.array(mesh.vc)
        return m


def cal_norm_scale(meshes, maxd=2.0):
    "compute the normalization scale"
    verts1 = []
    for m in meshes:
        verts1.append(m.v)
    verts1 = np.concatenate(verts1)
    bmin = np.min(verts1, 0)
    bmax = np.max(verts1, 0)
    scale = maxd/(bmax - bmin) # normalize to -1, 1

    return np.min(scale)

def get_faces_and_textures(verts_list, faces_list, colors_list=SMPL_OBJ_COLOR_LIST):
    """

    Args:
        verts_list (List[Tensor(B x V x 3)]).
        faces_list (List[Tensor(f x 3)]).

    Returns:
        faces: (1 x F x 3)
        textures: (1 x F x 1 x 1 x 1 x 3)
    """
    all_faces_list = []
    all_textures_list = []
    o = 0
    for verts, faces, colors in zip(verts_list, faces_list, colors_list):
        B = len(verts)
        index_offset = torch.arange(B).to(verts.device) * verts.shape[1] + o
        o += verts.shape[1] * B
        faces_repeat = faces.clone().repeat(B, 1, 1)
        faces_repeat += index_offset.view(-1, 1, 1)
        faces_repeat = faces_repeat.reshape(-1, 3)
        all_faces_list.append(faces_repeat)
        textures = torch.FloatTensor(colors).to(verts.device)
        all_textures_list.append(textures.repeat(faces_repeat.shape[0], 4, 4, 4, 1))
    all_faces_list = torch.cat(all_faces_list).unsqueeze(0)
    all_textures_list = torch.cat(all_textures_list).unsqueeze(0)
    return all_faces_list, all_textures_list


def get_kinect_K(image_size=2048):
    KINECT_SIZE = 2048.

    fx, fy = 979.784, 979.840  # for original kinect coordinate system
    cx, cy = 1018.952, 779.486

    ratio = image_size / KINECT_SIZE
    K = torch.cuda.FloatTensor([[[fx * ratio, 0, cx * ratio],
                                 [0, fy * ratio, cy * ratio],
                                 [0, 0, 1]]])
    return K, ratio


def setup_renderer(view='front', rotate=False, image_size=2048, ):
    K, ratio = get_kinect_K(image_size)
    w, h = 2048, 1536

    if view=='front':
        if rotate:
            R = torch.cuda.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
        else:
            R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        t = torch.zeros(1, 3).cuda()
    elif view=='top':
        theta = 1.3
        d = 1.3
        x, y = np.cos(theta), np.sin(theta)
        mx, my, mz = 0., 0., 2.5 # mean center
        R = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
        t = torch.cuda.FloatTensor([mx, my + d, mz])
    else:
        raise NotImplemented

    renderer = Renderer(
        image_size=image_size, K=K, R=R, t=t, orig_size=w * ratio
    )

    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.4 # if the person is far away, make this smaller
    renderer.background_color = [1, 1, 1]

    return renderer

def setup_side_renderer(dist=2.0, elev=45., azim=90., image_size=640):
    "to use this renderer, the meshes should be centered"
    renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    renderer.background_color = [1, 1, 1]

    renderer.eye = nr.get_points_from_angles(dist, elev, azim)
    renderer.light_direction = list(np.array(renderer.eye) / 2.2)
    return renderer


def align_to_input(crop_info, height, rend, train_crop_size, width, mean_cent=False, pad_value=255):
    """
    align rendered reconstruction with input image
    first crop and translate rendered image to match the crop center,
    then scale the crop to align with the input
    :param crop_info: a dict saved by test dataloader
    :param height: 1536
    :param rend: (2048, 2048) , rendered image
    :param train_crop_size: crop size at training time
    :param width: 2048
    :param mean_cent: the crop center is moved to mean center or not, False for BEHAVE and True for in the wild data
    :param pad_value: value used to pad boarders
    :return: input image overlapped with rendering
    """
    w, h = crop_info['rgb_newsize'] # test image resized to ~2048p size
    crop_center = crop_info['crop_center'].astype(int)

    # crop on recon projected image
    if mean_cent:
        mean_crop_center = np.array([1008, 995])
        top_left = mean_crop_center - train_crop_size // 2
        bottom_right = mean_crop_center + train_crop_size // 2
    else:
        top_left = crop_center - train_crop_size // 2
        bottom_right = crop_center + train_crop_size // 2

    pad_left = max(0, -top_left[0])
    pad_top = max(0, -top_left[1])
    pad_right = max(0, bottom_right[0] - width)
    pad_bottom = max(0, bottom_right[1] - height)
    top_left = np.maximum(np.zeros(2), top_left).astype(int)
    bottom_right = np.minimum(np.array([width, height]), bottom_right).astype(int)
    img_crop = rend[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if rend.ndim == 3:
        img_square = np.pad(img_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=pad_value)
    else:
        # mask only
        img_square = np.pad(img_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=pad_value)
    # resize to the crop in original image
    crop_size = int(crop_info['crop_size'][0])
    img_crop_orig = cv2.resize(img_square, (crop_size, crop_size))

    # now fit crop to original image
    # find the indexing in the original image
    if mean_cent:
        top_left = crop_center - crop_size // 2
        bottom_right = crop_center + (crop_size - crop_size // 2)
    else:
        top_left = crop_center - crop_size // 2
        bottom_right = crop_center + (crop_size - crop_size // 2)
    x1y1 = np.maximum(np.zeros(2), top_left).astype(int)
    x2y2 = np.minimum(np.array([w, h]), bottom_right).astype(int)

    # find the indexing in the cropped patch
    x1 = max(0, -top_left[0])
    y1 = max(0, -top_left[1])
    x2 = min(crop_size, crop_size - (bottom_right[0] - w))
    y2 = min(crop_size, crop_size - (bottom_right[1] - h))
    if rend.ndim == 3:
        overlap = np.zeros((h, w, 3)).astype(np.uint8) + pad_value
    else:
        overlap = np.zeros((h, w)).astype(np.uint8) + pad_value
    # feed back to original image
    overlap[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]] = img_crop_orig[y1:y2, x1:x2]

    return overlap


def load_mesh(pcfile):
    if not isfile(pcfile):
        return None
    m = Mesh()
    m.load_from_file(pcfile)
    return m