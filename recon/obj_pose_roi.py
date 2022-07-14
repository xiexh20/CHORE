"""
occlusion aware loss with rendering only the region of interest

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
import cv2
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import numpy as np
import neural_renderer as nr
from scipy.ndimage.morphology import distance_transform_edt
from detectron2.structures import BitMasks
from recon.opt_utils import mask2bbox
from recon.bbox import make_bbox_square, bbox_wh_to_xy, bbox_xy_to_wh


class SilLossROI(nn.Module):
    def __init__(self, person_masks,
                 obj_masks,
                 temp_mesh,
                 crop_centers,
                 rend_size=256, # phosa's render size
                 kernel_size=7,
                 bbox_expansion=0.3,
                 device='cuda:0'):
        """
        person masks and object masks are direct input to the network
        """
        super(SilLossROI, self).__init__()
        self.net_input_size = 512
        self.temp_mesh = temp_mesh  # the mesh shoud be centered
        B = person_masks.shape[0]

        # convert object mask to squared bbox
        obj_bboxes = self.masks2bboxes(obj_masks).astype(float) # xyxy format
        obj_bboxes_xywh = bbox_xy_to_wh(obj_bboxes)
        obj_bbox_squares = make_bbox_square(obj_bboxes_xywh, bbox_expansion) # xywh format
        obj_bbox_square_xyxy = bbox_wh_to_xy(obj_bbox_squares)
        obj_bbox_square_xyxy_th = torch.FloatTensor(obj_bbox_square_xyxy).to(device)

        # crop and resize mask to rend size
        bit_masks = BitMasks(obj_masks)
        obj_masks_crop = bit_masks.crop_and_resize(obj_bbox_square_xyxy_th, rend_size).clone().detach()
        # do the same for person mask ==>> find occlusion aware mask
        ps_bit_masks = BitMasks(person_masks)
        ps_masks_crop = ps_bit_masks.crop_and_resize(obj_bbox_square_xyxy_th, rend_size).clone().detach()

        scale = 1200/512. # crop size divided by network input size
        K_rois, keep_masks, image_refs = [],[],[]
        for ps, obj, bbox, crop_center in zip(ps_masks_crop, obj_masks_crop, obj_bbox_squares, crop_centers):
            mask = self.cvt_masks(ps, obj) # 1--foreground, -1: occlusion ignore
            image_refs.append((obj > 0).clone().float()) # for computing edges, just use the fore mask
            keep_masks.append(mask.clone().float()) # the rendered image is masked with this to occlusion-aware

            # convert bbox coordinate to original 1536px image coordinate
            bbox_orig = self.to_original_bbox(bbox, scale, crop_center.cpu().numpy())
            K = self.compute_K_roi(bbox_orig)
            K_rois.append(K)
        self.register_buffer("image_ref", torch.stack(image_refs, 0))
        self.register_buffer("keep_mask", torch.stack(keep_masks, 0))
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
        self.prepare_dist_trans(image_refs)  # disntance transform, on the object mask only
        cam_Ks = torch.cat(K_rois, 0)
        self.prepare_render(temp_mesh, B, cam_Ks, rend_size)

    def prepare_render(self, temp_mesh, batch_size, cam_Ks, rend_size):
        verts = torch.tensor(temp_mesh.v, dtype=torch.float32)
        faces = torch.tensor(temp_mesh.f)
        ts = 1
        textures = torch.ones(faces.shape[0], ts, ts, ts, 3, dtype=torch.float32).cuda()
        self.register_buffer("vertices", verts.repeat(batch_size, 1, 1))
        self.register_buffer("faces", faces.repeat(batch_size, 1, 1))
        self.register_buffer("textures", textures.repeat(batch_size, 1, 1, 1, 1, 1))
        R = torch.eye(3).unsqueeze(0).cuda()
        t = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=rend_size,
            K=cam_Ks,
            R=R,
            t=t,
            orig_size=1,
            anti_aliasing=False,
        )

    def prepare_dist_trans(self, image_refs, power=0.25):
        "compute distance transform"
        ref_edges = []
        for ref in image_refs:
            mask_edge = self.compute_edges(ref.unsqueeze(0)).cpu().numpy()  # edges of the silhoutte
            edt = distance_transform_edt(1 - (mask_edge > 0)) ** (power * 2)
            ref_edges.append(edt)
        ref_edges = np.concatenate(ref_edges, 0)
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(ref_edges).float()
        )

    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    @staticmethod
    def to_original_bbox(bbox_square, scale, trans, crop_size=1200):
        """
        bbox_square: bbox in local path
        scale: 1200/512
        """
        bbox_orig = bbox_square.copy()
        bbox_orig *= scale # to 1200 square
        bbox_orig[:2] += trans - crop_size / 2.0 # coordinate in original image
        return bbox_orig

    @staticmethod
    def compute_K_roi(bbox_square, kinect_width=2048):
        """
        render image with a focus on region of interest, use kinect camera parameters
        """
        x, y, b, w = bbox_square
        assert b == w, "the given bbox is not square!"

        fx, fy = 979.7844/kinect_width, 979.840/kinect_width
        cx, cy = 1018.952/kinect_width, 779.486/kinect_width

        fx_ = fx * kinect_width / b
        fy_ = fy * kinect_width / b
        cx_ = (cx * kinect_width - x)/b
        cy_ = (cy * kinect_width - y)/b

        K = torch.cuda.FloatTensor([[[fx_, 0, cx_], [0, fy_, cy_], [0, 0, 1]]])
        return K

    def cvt_masks(self, person_mask, obj_mask):
        """
        following phosa to convert mask for occlusion aware loss
        keep_masks contains 0: background, 1: foreground, and -1: occlusion, i.e. mask of another instance
        :return keep mask for occlusion aware
        """
        fore_mask = obj_mask > 0.5
        ps_mask = person_mask > 0.5
        mask_merge = ps_mask | fore_mask
        # bkg_mask = ~mask_merge
        mask_inv = - ps_mask.clone().float()
        mask_inv[fore_mask] = 1.  # convention: 1--foreground, -1: occlusion ignore
        # return mask_inv >= 0
        return mask_inv >= 0 # both foreground and fully background are taken into account

    @staticmethod
    def masks2bboxes(masks):
        """
        input: torch tensor (B, W, H), range (0,1), return numpy (B, 4) in xyxy format
        """
        bboxes = []
        for mask in masks:
            bbox = mask2bbox((mask.cpu().numpy()*255).astype(np.uint8))
            bboxes.append(bbox)
        return np.stack(bboxes, 0)

    def forward(self, R, obj_t, obj_s):
        """
        transform object template, render masks and compute loss
        :param R: (B, 3, 3)
        :param obj_t: (B, 3)
        :param obj_s: (B, )
        :return: loss dict and others used for debug
        """
        verts = self.apply_transformation(R, obj_t, obj_s)
        image = self.keep_mask * self.renderer(verts, self.faces, mode="silhouettes")  # first mask the rendered image using GT mask
        loss_dict = {}
        loss_dict["mask"] = torch.sum((image - self.image_ref) ** 2, dim=(1, 2)).mean()  # L2 mask silhouette
        edges = self.compute_edges(image)
        return loss_dict, image, edges, self.image_ref, self.edt_ref_edge

    def apply_transformation(self, R, obj_t, obj_s):
        verts = torch.bmm(self.vertices, R) + obj_t.unsqueeze(1)
        verts = obj_s.view(-1, 1, 1)*verts
        return verts

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means xy between [-1, 1] and far > depth > 0
        proj = nr.projection(
            verts,
            self.renderer.K,
            self.renderer.R,
            self.renderer.t,
            self.renderer.dist_coeffs,
            orig_size=self.renderer.orig_size,
        )
        xy, z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(z)
        lower_right = torch.max(xy - 1, zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - xy, zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-z, zeros).sum(dim=(1, 2))
        too_far = torch.max(z - self.renderer.far, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far
