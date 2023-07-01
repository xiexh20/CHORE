"""
base class for joint reconstruction optimization
if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
import cv2
sys.path.append(os.getcwd())
import torch
import pickle as pkl
from os.path import join, isfile
import torch.nn.functional as F
import json
import numpy as np
import trimesh
from psbody.mesh.sphere import Sphere
from psbody.mesh import Mesh, MeshViewer
from sklearn.decomposition import PCA

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss

from lib_smpl.smpl_generator import SMPLHGenerator
from behave.seq_utils import SeqInfo
from lib_smpl.th_hand_prior import HandPrior
from lib_smpl.th_smpl_prior import get_prior
from lib_smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from lib_smpl.const import SMPL_POSE_PRAMS_NUM, SMPL_PARTS_NUM

from model.chore import CHORE
from model.camera import KinectColorCamera
import recon.opt_utils as opt_utils

import yaml, sys
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
SMPL_ASSETS_ROOT = paths["SMPL_ASSETS_ROOT"]
BEHAVE_PATH = paths['BEHAVE_PATH']
RECON_PATH = paths['RECON_PATH']


class ReconFitterBase:
    def __init__(self, seq_folder,
                 device='cuda:0', debug=False,
                 obj_name=None,
                 outpath=RECON_PATH,
                 args=None):
        # obtain object scan and canonical pose
        self.seq_folder = seq_folder
        self.outpath = outpath
        self.device = device
        if not isfile(seq_folder + '/info.json'):
            assert obj_name is not None, 'must provide the name of the object to be reconstructed!'
            # use default settings
            self.gender = 'male'
        else:
            # load object name and gender information from sequence information file
            self.seq_info = SeqInfo(seq_folder)
            obj_name = self.seq_info.get_obj_name()
            self.gender = self.seq_info.get_gender()
        pca_init, obj_points = self.compute_pca_init(obj_name)
        self.pca_init = torch.tensor(pca_init, dtype=torch.float32).to(device)
        self.obj_points = torch.tensor(obj_points, dtype=torch.float32).to(device)
        self.obj_scale = 1.0  # default scale for the object
        self.part_labels = self.load_part_labels()  # (6890, )

        # network configurations
        self.camera = KinectColorCamera(args.loadSize)
        self.net_in_size = args.net_img_size[0]  # network input image size
        self.z_0 = 2.2 if 'z_0' not in args else args.z_0  # fixed smpl center depth

        # for collision loss
        self.scan_faces = torch.tensor(self.scan.f).long().to(self.device)
        sigma = 0.5
        point2plane = False
        max_collisions = 8
        self.search_tree = BVH(max_collisions=max_collisions)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                                         point2plane=point2plane,
                                                                         vectorized=True)

        # Logging
        self.mv = MeshViewer(window_width=500, window_height=500,) if debug else None
        self.debug = debug
        self.part_names={
            0:'head',
            1:'left foot',
            2:'left hand',
            3:'left leg',
            4:'left midarm',
            5:'left upper arm',
            6:'right foot',
            7:'right hand',
            8:'right leg',
            9:'right midarm',
            10:'right right upper arm',
            11:'torso',
            12:'upper left leg',
            13:'upper right leg'
        }

    def compute_pca_init(self, obj_name):
        """
        load object template, and compute canonical pose
        :param obj_name: name of the object template
        :return:
        """
        scan = opt_utils.load_scan_centered(opt_utils.get_template_path(BEHAVE_PATH+"/../objects", obj_name))
        scan.v = scan.v - np.mean(scan.v, 0)
        self.scan = scan
        obj_verts = scan.v
        pca = PCA(n_components=3)
        pca.fit(obj_verts)
        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=scan.f, process=False)
        x = obj_mesh.sample(3000)
        return pca.components_, x

    def get_smpl_init(self, image_paths, trans):
        """
        load FrankMocap pose prediction and initialize a SMPL-H model
        :param image_paths: input test image paths
        :param trans: smpl translation init
        :return: SMPLH model 
        """
        mocap_paths = [x.replace(".color.jpg", '.mocap.json') for x in image_paths]
        # load mocap data
        mocap_poses, mocap_betas = [], []
        for file in mocap_paths:
            p, b = self.load_mocap(file)
            mocap_betas.append(b)
            mocap_poses.append(p)
        smpl = SMPLHGenerator.get_smplh(np.stack(mocap_poses, 0),
                              np.stack(mocap_betas, 0),
                              trans, self.gender)
        return smpl

    def load_mocap(self, file):
        """
        load SMPL pose and shape predicted by FrankMocap
        :param file:
        :return: FrankMocap predicted pose (72, ) and shape (10, ) parameters
        """
        params = json.load(open(file))
        pose_local = np.array(params['pose'])
        beta = np.array(params['betas'])
        return pose_local, beta

    @staticmethod
    def init_object_orientation(tgt_axis, src_axis):
        """
        given orientation of template mesh, find the relative transformation
        :param tgt_axis: target object PCA axis
        :param src_axis: object template PCA axis
        :return: relative rotation from template to target
        """
        pseudo = ReconFitterBase.inverse(src_axis)
        rot = torch.bmm(pseudo, tgt_axis)

        return ReconFitterBase.decopose_axis(rot)

    @staticmethod
    def project_so3(mat):
        """
        project 3x3 matrix to SO(3) real
        Args:
            mat: (B, 3, 3)

        Returns: (B, 3, 3) real rotation matrix
        References: https://github.com/amakadia/svd_for_pose

        this does: US'V^T, where S'=diag(1, ..., 1, det(UV^T)), symmetric orthogonalization that project a matrix to SO(3)
        however, this operation is not orientation preserving when det(UV^T)<=0

        """
        assert mat.shape[1:] == (3, 3), f'invalid shape {mat.shape}'
        u, s, v = torch.svd(mat)
        vt = torch.transpose(v, 1, 2)
        det = torch.det(torch.matmul(u, vt))
        det = det.view(-1, 1, 1)
        vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
        r = torch.matmul(u, vt)
        return r

    @staticmethod
    def inverse(mat):
        assert len(mat.shape) == 3
        tr = torch.bmm(mat.transpose(2, 1), mat)
        tr_inv = torch.inverse(tr)
        inv = torch.bmm(tr_inv, mat.transpose(2, 1))
        return inv

    def fit_recon(self, args):
        raise NotImplemented

    def optimize_smpl(self, smpl, data_dict, iter_for_betas=10, iter_for_pose=10,
                      iter_for_kpts=5, steps_per_iter=10, max_iter=100):
        """
        optimize SMPL parameters to fit to neural reconstruction
        :param smpl: SMPL model
        :param data_dict:
        :param iter_for_betas: number of iterations to optimize betas and global pose
        :param iter_for_pose: number of iterations to optimize SMPL with neural predictions
        :param iter_for_kpts: number of iterations to optimize SMPL with neural predictions + 2D keypoints
        :param steps_per_iter: how many steps for one iteration
        :param max_iter: maximum iterations run before until convergence
        :return:
        """
        raise NotImplemented

    def forward_smpl(self, smpl, data_dict, phase):
        """
        one forward step for SMPL optimization
        :param smpl:
        :param data_dict:
        :param phase:
        :return:
        """
        raise NotImplemented

    def forward_step(self, model, smpl, data_dict, obj_R, obj_t, obj_s, phase):
        """one forward step for jointly optimize SMPL and object"""
        raise NotImplemented

    def smplz_loss(self, J, loss_dict):
        loss_dict['smplz'] = torch.mean((J[:, 8, 2] - self.z_0) ** 2)  # the smpl depth is fixed

    def is_done(self, image_paths, save_name, test_id):
        smpl_files, obj_files = self.get_output_paths(image_paths, save_name, test_id)
        for sf, of in zip(smpl_files, obj_files):
            if not isfile(sf) or not isfile(of):
                return False
        return True

    def get_output_paths(self, image_paths, save_name, test_id):
        """

        :param image_paths: test image file path:  ROOT/SEQ_NAME/frame_time/kx.color.jpg
        :param save_name: subfolder name for the reconstruction results
        :param test_id: test on images from which kinect, default 1
        :return: the output paths of the reconstructed SMPL and object files
        """
        seq_names = [x.split(os.sep)[-3] for x in image_paths]
        frame_times = [x.split(os.sep)[-2] for x in image_paths]
        outfolders = [join(self.outpath, seq, frame, save_name) for seq, frame in zip(seq_names, frame_times)]
        obj_files, smpl_files = [], []
        for folder in outfolders:
            os.makedirs(folder, exist_ok=True)
            smpl_files.append(join(folder, f'k{test_id}.smpl.ply'))
            obj_files.append(join(folder, f'k{test_id}.object.ply'))
        return smpl_files, obj_files

    def save_outputs(self, smpl, obj_R, obj_t, traindata_paths, save_name, test_id, obj_s=None):
        smpl_files, obj_files = self.get_output_paths(traindata_paths, save_name, test_id)
        opt_utils.save_smplfits(smpl_files, np.zeros(len(obj_t)), smpl)

        B = len(obj_files)
        obj_verts = torch.tensor(self.scan.v, dtype=torch.float32).repeat(B, 1, 1).to(self.device)
        obj_verts = self.transform_object(obj_verts, obj_R, obj_t, obj_s)

        obj_R = self.decopose_axis(obj_R, no_rand=True) # save rotation matrix
        verts = obj_verts.detach().cpu().numpy()
        for v, p, r, s, t in zip(verts, obj_files, obj_R.detach().cpu().numpy(), obj_s.detach().cpu().numpy(), obj_t.detach().cpu().numpy()):
            m = Mesh(v=v, f=self.scan.f)
            m.write_ply(p)

            param_file = p.replace(".ply", '.pkl')
            pkl.dump({
                'rot':r, "trans":t, 'scale':s
            }, open(param_file, 'wb'))

    def load_part_labels(self):
        """
        part labels for each SMPL vertex
        :return: (6890, )
        """
        part_labels = pkl.load(open(SMPL_ASSETS_ROOT+'/smpl_parts_dense.pkl', 'rb'))
        labels = np.zeros((6890,), dtype='int32')
        for n, k in enumerate(part_labels):
            labels[part_labels[k]] = n
        labels = torch.tensor(labels)
        return labels.to(self.device)

    def load_part_labels_batch(self, batch_size):
        labels = self.load_part_labels()
        labels = labels.repeat(batch_size, 1).to(self.device)
        return labels.long()

    def get_body_kpts2d(self, traindata_paths, tol=0.3):
        "load 25 body keypoints detected by openpose"
        json_paths = self.get_kpt_paths(traindata_paths)
        return self.load_kpts(json_paths, tol)

    def get_kpt_paths(self, image_paths):
        json_paths = [x.replace('.color.jpg', '.color.json') for x in image_paths]
        return json_paths

    def load_kpts(self, json_paths, tol):
        """
        load 2d body keypoints, in original image space (before crop and scaling etc.)
        :param json_paths:
        :param tol:
        :return: (B, N, 3), xy coordinate of the keypoints and confidence
        """
        kpts = []
        for file in json_paths:
            data = json.load(open(file))
            J2d = np.array(data["body_joints"]).reshape((-1, 3))
            J2d[:, 2][J2d[:, 2] < tol] = 0
            kpts.append(J2d)
        kpts = np.stack(kpts, 0)
        return torch.tensor(kpts, dtype=torch.float32).to(self.device)

    def scale_body_kpts(self, kpts, resize_scale, crop_scale, crop_center):
        """
        get the kpts coordinate in network input image
        kpts: (B, 25, 3) 2d kpts in original image coordinate before any resizing, crop and scaling
        crop center: (B, 2), the crop center used for cropping image
        resize scale: (B, ) image resizing factor to make the image 2048 p
        crop scale: (B, )cropping scale factor to make the person depth close to 2.2m
        """
        pxy = kpts[:, :, :2] * resize_scale.unsqueeze(1).unsqueeze(1) # coordinate in 2048p image
        crop_size_org = crop_scale * self.camera.crop_size # actual cropping size happened in 2048p image (B,)
        pxy = pxy - crop_center.unsqueeze(1) + crop_size_org.unsqueeze(1).unsqueeze(1) / 2   # coordinate in cropped image
        pxy = pxy * self.net_in_size / crop_size_org.unsqueeze(1).unsqueeze(1) # coordinate in network input image
        return torch.cat([pxy, kpts[:, :, 2:3]], -1)

    def get_loss_weights(self):
        raise NotImplemented

    def optimize_smpl_object(self, model:CHORE, data_dict, obj_iter=20, joint_iter=10, steps_per_iter=10):
        """
        jointly optimize SMPL together with the object
        :param model:
        :param data_dict:
        :return:
        """
        raise NotImplemented

    def get_loss_str(self, it, loss_dict, weight_dict, weight_decay):
        l_str = 'Iter: {}'.format(it)
        for k in loss_dict:
            l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], weight_decay).mean().item())
        return l_str

    @staticmethod
    def sum_dict(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def transform_object(self, object_init, rot, obj_t, obj_s):
        "project 3x3 matrix to SO(3) and then apply transformation to template object"
        R = self.decopose_axis(rot)
        object = self.transform_obj_verts(object_init, R, obj_t, obj_s)
        return object

    def transform_obj_verts(self, verts, obj_R, obj_t, obj_s):
        "do scale after rotation and translation"
        verts = torch.bmm(verts, obj_R) + obj_t.unsqueeze(1)
        verts = verts * obj_s.unsqueeze(1).unsqueeze(1)
        return verts

    @staticmethod
    def decopose_axis(rot, no_rand=False):
        """
        project 3x3 matrix to SO(3)
        :param rot: (B, 3, 3), predicted/optimized rotation matrix
        :param no_rand: add random noise to the matrix (prevent SVD divengence) or not
        :return: (B, 3, 3) rotation matrix in SO(3)
        """
        if no_rand:
            return ReconFitterBase.project_so3(rot)
        else:
            return ReconFitterBase.project_so3(rot + 1e-4 * torch.rand(rot.shape[0], 3, 3).to(rot.device))

    def prepare_query_dict(self, batch):
        """
        dict of additional data required for query besides query points
        :param batch: a batch of data from dataloader
        :return:
        """
        crop_center = batch.get('crop_center').to(self.device)  # (B, 3)
        ret = {
            'crop_center': crop_center,
        }
        return ret

    def prep_smplfit(self, data, generator, pc_generated):
        """
        prepare data dict for SMPL optimization
        :param data: a batch of data from dataloader
        :param generator: PointCloud generator
        :param pc_generated:
        :return:
        """
        batch_size = data['images'].shape[0]
        human_points = pc_generated['human']['points'].clone().detach().to(self.device)
        obj_points = pc_generated['object']['points'].clone().detach().to(self.device)
        human_parts = pc_generated['human']['parts'].clone().detach().to(self.device)
        print('Points generation done, in total {} human points, {} object points'.format(human_points.shape[1],
                                                                                          obj_points.shape[1]))

        # use predicted human centers
        human_t = pc_generated['human']['centers'][:, :3]

        human_t[:, 2] = self.z_0  # fix smpl depth
        smpl = self.get_smpl_init(data.get('path'), human_t)

        part_labels = self.load_part_labels_batch(batch_size)
        part_colors = self.get_parts_colors(pc_generated['human']['parts'])
        body_kpts = self.get_body_kpts2d(data.get('path'))
        body_kpts = self.scale_body_kpts(body_kpts,
                                         data.get("resize_scale").to(self.device),
                                         data.get("crop_scale").to(self.device),
                                         data.get('old_crop_center').to(self.device)
                                         ) # preprocess body keypoints
        body_kpts = body_kpts.clone().float().to(self.device)
        query_dict = self.prepare_query_dict(data)
        betas_dict = {
            'images': data.get('images').to(self.device),
            'human_init': human_points,
            'human_parts': human_parts,
            'part_labels': part_labels,
            'part_colors': part_colors,
            'body_kpts': body_kpts,
            'query_dict': query_dict,
            'net': generator.model,
            'pose_init': smpl.pose[:, 3:SMPL_POSE_PRAMS_NUM].clone().to(self.device)
        }
        return betas_dict, body_kpts, human_parts, human_points, human_t, obj_points, part_colors, part_labels, query_dict, smpl

    def visualize_fitting(self, data_dict, object, smpl, smpl_verts):
        idx = 0
        input_image = data_dict['images'][idx].cpu().numpy().transpose((1, 2, 0))
        # visualize keypoints
        J, face, hands = smpl.get_landmarks()
        crop_center = data_dict['query_dict']['crop_center'] if 'crop_center' in data_dict['query_dict'] else None
        pxy = self.project_points(J, crop_center)
        px, py = pxy[:, :, 0], pxy[:, :, 1]
        img_vis = cv2.cvtColor((input_image[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        # kstr = "kpts: "
        for i, (x, y) in enumerate(zip(px[0], py[0])):
            cv2.circle(img_vis, (int(x), int(y)), 2, (0, 255, 255), 2, cv2.LINE_8)
        # visualize gt kpts as well
        if 'body_kpts' in data_dict:
            kpts_gt = data_dict['body_kpts'][0][:, :2].cpu().numpy()
            for p in kpts_gt:
                loc = (int(p[0]), int(p[1]))
                cv2.circle(img_vis, loc, 2, (0, 0, 255), 1, cv2.LINE_8)
        cv2.imshow('input image', img_vis)
        cv2.moveWindow('input image', 30, 0)
        cv2.waitKey(10)
        obj = Mesh(object[idx].detach().cpu().numpy(), [], vc='red')
        if 'obj_part_labels' in data_dict:
            po = torch.argmax(data_dict['obj_part_labels'], 1)
            po = po[idx].detach().cpu().numpy()
            # print(po)
            obj.set_vertex_colors_from_weights(po)

        # obj_init = Mesh(data_dict['obj_init'][idx].detach().cpu().numpy(), [], vc='yellow')
        obj_init = Mesh(data_dict['obj_init'][idx].detach().cpu().numpy(), [], vc='yellow')
        hum = Mesh(smpl_verts[idx].detach().cpu().numpy(), [], vc='green')
        hum_init = Mesh(data_dict['human_init'][idx].detach().cpu().numpy(), [], vc=data_dict['part_colors'])
        # visualize parts
        hum_init.set_vertex_colors_from_weights(data_dict['human_parts'][idx].detach().cpu().numpy())

        meshes = [obj, obj_init, hum, hum_init]
        if 'contact_mask_h' in data_dict:
            # visualize verts where there is contact
            mask = data_dict['contact_mask_h'][idx]
            if torch.sum(mask) > 0:
                contact = Mesh(smpl_verts[idx, mask].detach().cpu().numpy(), [], vc='blue')
                meshes.append(contact)
        if 'contact_mask_o' in data_dict:
            # visualize verts where there is contact
            mask = data_dict['contact_mask_o'][idx]
            if torch.sum(mask) > 0:
                contact = Mesh(object[idx, mask].detach().cpu().numpy(), [], vc='blue')
                meshes.append(contact)
        if 'obj_center_pred' in data_dict:
            sphere = Sphere(data_dict['obj_center_pred'][idx].detach().cpu().numpy(), 0.06).to_mesh((0, 1.0, 1.0))
            meshes.append(sphere)
        if 'smpl_center_pred' in data_dict:
            sphere = Sphere(data_dict['smpl_center_pred'][idx].detach().cpu().numpy(), 0.06).to_mesh((1., 1.0, 0)) # yellow
            meshes.append(sphere)
        if 'smpl_center_act' in data_dict:
            sphere = Sphere(data_dict['smpl_center_act'][idx].detach().cpu().numpy(), 0.06).to_mesh((1., 0, 1.0))
            meshes.append(sphere)

        if 'neigh_o' in data_dict:
            meshes.append(data_dict['neigh_o'])

        # add gt mesh
        if 'hum_gt' in data_dict:
            hum_gt = Mesh(data_dict['hum_gt'].verts_list()[idx].detach().cpu().numpy(),
                          data_dict['hum_gt'].faces_list()[idx].cpu().numpy())
            obj_gt = Mesh(data_dict['obj_gt'].verts_list()[idx].cpu().numpy(),
                              data_dict['obj_gt'].faces_list()[idx].cpu().numpy())
            meshes.extend([hum_gt, obj_gt])

        self.mv.set_dynamic_meshes(meshes)

    def compute_obj_loss(self, data_dict, loss_dict, model, obj_s, object):
        "object df_o loss"
        model.query(object, **data_dict['query_dict'])
        preds = model.get_preds()
        df_pred, pca_pred, parts_pred = preds[:3]
        loss_dict['object'] = torch.clamp(df_pred[:, 1:2, :], max=0.8).mean()
        loss_dict['scale'] = torch.mean((obj_s - self.obj_scale) ** 2)
        return preds

    def compute_prior_loss(self, loss_dict, smpl, nobeta=False):
        """
        SMPL prior loss
        :param loss_dict:
        :param smpl:
        :param nobeta: compute beta prior loss or not
        :return:
        """
        prior = get_prior()
        if not nobeta:
            loss_dict['beta'] = torch.mean(smpl.betas ** 2)
        loss_dict['pose'] = torch.mean(prior(smpl.pose[:, :72]))
        hand_prior = HandPrior(type='grab')
        loss_dict['hand'] = torch.mean(hand_prior(smpl.pose))

    def compute_df_h_loss(self, data_dict, loss_dict, model, smpl_verts):
        "human df_h loss"
        model.query(smpl_verts, **data_dict['query_dict'])
        df_pred, pca_pred, parts_pred, centers_pred = model.get_preds()
        loss_dict['df_h'] = torch.clamp(df_pred[:, 0:1, :], max=0.1).mean()
        return df_pred, parts_pred, centers_pred

    def compute_smpl_center_pred(self, data_dict, model, smpl):
        """compute SMPL center predictions for SMPL verts"""
        with torch.no_grad():
            smpl_verts, _, _, _ = smpl()
            model.query(smpl_verts, **data_dict['query_dict'])
            df_pred, pca_pred, parts_pred, centers_pred = model.get_preds()
            pred = torch.mean(centers_pred[:, :3], -1)
            return pred

    def compute_contact_loss(self, df_hum_o, df_obj_h, object, smpl_verts,
                             loss_dict,
                             part_o=None):
        """
        pull contact points together
        :param df_hum_o: (B, N_h) object distance predicted for smpl verts
        :param df_obj_h: (B, N_o) human distance predicted for object verts
        :param object: (B, N_o, 3) all object surface points
        :param smpl_verts: (B, N_h, 3) all SMPL vertices
        :param loss_dict:
        :param part_o: part labels predicted for object verts
        :return:
        """
        # contact_loss = []
        contact_mask_o = df_obj_h < 0.08  # contacts on object points
        contact_mask_h = df_hum_o < 0.08  # contact on human verts
        contact_names = []
        contact_points_h, contact_points_o = [], []
        part_o = torch.argmax(part_o, 1)
        for hum, obj, mh, mo, po in zip(smpl_verts, object, contact_mask_h, contact_mask_o, part_o):
            # iterate each example
            ch, co = torch.sum(mh), torch.sum(mo) # count of contact vertices
            if ch + co == 0:
                continue  # contact not found on both human and object verts
            if co > 0:
                obj_v = obj[mo] # object vertices in contact
                label_o = po[mo]
            else:
                obj_v = obj  # pull all object verts to the human
                label_o = po
            if ch > 0:
                hum_v = hum[mh]
                label_h = self.part_labels[mh]
            else:
                hum_v = hum  # pull all smpl verts to the object
                label_h = self.part_labels

            # find pairs based on part labels
            for i in range(SMPL_PARTS_NUM):
                if i not in label_h or i not in label_o:
                    continue
                hum_ind = torch.where(label_h == i)[0]
                obj_ind = torch.where(label_o == i)[0]
                hp = hum_v[hum_ind] # human points in contact
                op = obj_v[obj_ind] # object points in contact
                contact_points_h.append(hp)
                contact_points_o.append(op)
                contact_names.append(self.part_names[i])
        if len(contact_points_o) == 0:
            print('no contact')
            return
        # pull contact points together
        pc_h = Pointclouds(contact_points_h)
        pc_o = Pointclouds(contact_points_o)
        dist, _ = chamfer_distance(pc_h, pc_o)
        loss_dict['contact'] = dist

    def smpl_obj_collision(self, smpl_verts, smpl_faces, obj_verts, obj_faces):
        comb_verts = torch.cat([smpl_verts, obj_verts], 1)  # (B, N, 3)
        smpl_verts_count = smpl_verts.shape[1]
        comb_faces = torch.cat([smpl_faces, obj_faces + smpl_verts_count], 0)  # (F, 3)

        bs, nv = comb_verts.shape[:2]
        faces_idx = comb_faces + \
                    (torch.arange(bs, dtype=torch.long).to(self.device) * nv)[:, None, None]
        triangles = comb_verts.reshape([-1, 3])[faces_idx]

        with torch.no_grad():
            # find out the collision index
            collision_idxs = self.search_tree(triangles)
        pen_loss = torch.mean(self.pen_distance(triangles, collision_idxs))
        return pen_loss

    def compute_collision_loss(self, smpl_verts, smpl_faces, obj_R, obj_t, obj_s):
        """
        collision loss between SMPL and object meshes
        :param smpl_verts:
        :param smpl_faces:
        :param obj_R:
        :param obj_t:
        :param obj_s:
        :return:
        """
        scan_verts = torch.tensor(self.scan.v, dtype=torch.float32).repeat(obj_R.shape[0], 1, 1).to(self.device)
        verts = self.transform_obj_verts(scan_verts, obj_R, obj_t, obj_s)
        pen_loss = self.smpl_obj_collision(smpl_verts, smpl_faces, verts, self.scan_faces)
        return pen_loss

    def compute_kpts_loss(self, data_dict, loss_dict, smpl):
        """
        2D body keypoint loss
        :param data_dict:
        :param loss_dict:
        :param smpl:
        :return:
        """
        J, face, hands = smpl.get_landmarks()
        loss_dict['j2d'] = self.projection_loss(J, data_dict['body_kpts'], data_dict['query_dict']['crop_center'])

    def get_parts_colors(self, part):
        "part: (B, N), tensor"
        from recon.opt_utils import mturk_colors
        colors = np.zeros((*part.shape[:2], 3))
        for i in range(14):
            mask = part == i
            colors[mask, :] = mturk_colors[i]
        return colors

    def project_points(self, joints3d, crop_center=None):
        """
        project 3D points to the local coordinate of the scaled image path
        :param joints3d: 3D body keypoints
        :param crop_center: crop center in the original 1536x2048 image
        :return: 2D body keypoints, in the network input image space
        """
        px, py = self.camera.project_screen(joints3d, crop_center) # project to the cropped patch
        joints_proj = torch.cat([px, py], -1) * self.net_in_size / self.camera.crop_size # scale to input image space
        return joints_proj

    def projection_loss(self, joints3d, joints2d, crop_center):
        joints_proj = self.project_points(joints3d, crop_center)
        loss = F.mse_loss(joints_proj[:, :, :2], joints2d[:, :, :2], reduction='none')
        sum_loss = torch.mean(torch.sum(loss, axis=-1) * joints2d[:, :, 2])
        return sum_loss

    def split_smpl(self, smpl):
        split = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl)
        return split

    def copy_smpl_params(self, split_smpl, smpl):
        smpl.pose.data[:, :3] = split_smpl.global_pose.data
        smpl.pose.data[:, 3:66] = split_smpl.body_pose.data
        smpl.pose.data[:, 66:] = split_smpl.hand_pose.data
        smpl.betas.data[:, :2] = split_smpl.top_betas.data

        smpl.trans.data = split_smpl.trans.data

        return smpl

    def get_smpl_bbox(self, smpl):
        "obtain 3D smpl bounding boxes"
        verts, _, _, _ = smpl()
        bmin, _ = torch.min(verts, 1)
        bmax, _ = torch.max(verts, 1)
        return bmin, bmax

    def get_smpl_height(self, smpl):
        bmin, bmax = self.get_smpl_bbox(smpl)
        bbox = bmax - bmin
        return bbox[:, 1]

    def save_neural_recon(self, train_paths, recon_batch, save_name, tid):
        "save a batch of neural reconstruction"
        for i, x in enumerate(train_paths):
            seq, frame = str(x).split(os.sep)[-3], str(x).split(os.sep)[-2]
            folder = join(self.outpath, seq, frame, save_name)
            os.makedirs(folder, exist_ok=True)
            npz_file = join(folder, f'k{tid}_densepc.npz')
            out_dict = {}
            for tar in recon_batch:
                tar_i = {}
                for t in recon_batch[tar]:
                    tar_i[t] = recon_batch[tar][t][i].cpu().numpy()
                out_dict[tar] = tar_i
            np.savez(npz_file, **out_dict)
            print("{} saved".format(npz_file))

    def init_obj_fit_data(self, batch_size, human_t, pc_generated, scale):
        """
        initialize object fits and prepare data for optimization
        :param batch_size:
        :param human_t: optimized human body center
        :param pc_generated:
        :param scale: initial object scale
        :return: object rotation, scale, translation and surface points sampled from canonical object template
        """
        # use predicted object center
        obj_t = pc_generated['object']['centers'][:, 3:].to(self.device) + human_t.to(self.device)  # obj_t is relative to smpl center
        obj_t = obj_t.clone().detach().to(self.device)
        obj_t = obj_t.requires_grad_(True)

        pca_axis = pc_generated['object']['pca_axis'].to(self.device)
        pca_axis_init = torch.stack([self.pca_init for x in range(batch_size)], 0)
        obj_R = self.init_object_orientation(pca_axis, pca_axis_init)

        # baseline: no pca axis init
        # r = torch.eye(3).repeat(batch_size, 1, 1).to(self.device)
        # obj_R = torch.tensor(r).to(self.device)

        obj_R = obj_R.requires_grad_(True)
        # obj_s = scale.clone().float().to(self.device)
        obj_s = scale.clone().detach().to(self.device)
        obj_s = obj_s.requires_grad_(True)
        object_init = torch.stack([self.obj_points for x in range(batch_size)], 0)
        return obj_R, obj_s, obj_t, object_init

    def visualize_smpl_fit(self, data_dict, smpl, smpl_verts):
        idx = 0
        # visualize keypoints
        input_image = data_dict['images'][idx].cpu().numpy().transpose((1, 2, 0))
        J, face, hands = smpl.get_landmarks()
        crop_center = data_dict['query_dict']['crop_center'] if 'crop_center' in data_dict['query_dict'] else None
        pxy = self.project_points(J, crop_center)
        px, py = pxy[:, :, 0], pxy[:, :, 1]
        img_vis = cv2.cvtColor((input_image[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        kstr = "kpts: "
        for i, (x, y) in enumerate(zip(px[0], py[0])):
            cv2.circle(img_vis, (int(x), int(y)), 2, (0, 255, 255), 2, cv2.LINE_8)
        # visualize gt kpts as well
        if 'body_kpts' in data_dict:
            kpts_gt = data_dict['body_kpts'][0][:, :2].cpu().numpy()
            for p in kpts_gt:
                loc = (int(p[0]), int(p[1]))
                cv2.circle(img_vis, loc, 2, (0, 0, 255), 1, cv2.LINE_8)
        # d = 1
        cv2.imshow('input image', img_vis)
        cv2.waitKey(10)
        cv2.moveWindow('input image', 30, 0)

        hum = Mesh(smpl_verts[idx].detach().cpu().numpy(), [], vc='green')
        hum_init = Mesh(data_dict['human_init'][idx].detach().cpu().numpy(), [], vc=data_dict['part_colors'])
        # visualize parts
        hum_init.set_vertex_colors_from_weights(data_dict['human_parts'][idx].detach().cpu().numpy())
        meshes = [hum, hum_init]
        if 'contact_mask' in data_dict:
            # visualize verts where there is contact
            mask = data_dict['contact_mask'][idx]
            if torch.sum(mask) > 0:
                contact = Mesh(smpl_verts[idx, mask].detach().cpu().numpy(), [], vc='blue')
                meshes.append(contact)
            else:
                pass
        if 'smpl_center_pred' in data_dict:
            sphere = Sphere(data_dict['smpl_center_pred'][idx].detach().cpu().numpy(), 0.05).to_mesh((1.0, 1.0, 0))
            meshes.append(sphere)
        # add gt mesh
        if 'hum_gt' in data_dict:
            hum_gt = Mesh(data_dict['hum_gt'].verts_list()[idx].detach().cpu().numpy(),
                          data_dict['hum_gt'].faces_list()[idx].cpu().numpy())
            obj_gt = Mesh(data_dict['obj_gt'].verts_list()[idx].cpu().numpy(),
                          data_dict['obj_gt'].faces_list()[idx].cpu().numpy())
            meshes.extend([hum_gt, obj_gt])
        self.mv.set_dynamic_meshes(meshes)

    def visualize_contact_fitting(self, data_dict, edges, image, model, obj_center_pred, object, smpl, smpl_verts):
        with torch.no_grad():
            model.query(smpl_verts, **data_dict['query_dict'])
            df_pred, pca_pred, parts_pred = model.get_preds()[:3]
            contact_mask = df_pred[:, 1, :] < 0.08
            data_dict['contact_mask_h'] = contact_mask

            model.query(object, **data_dict['query_dict'])
            df_pred, pca_pred, parts_pred, centers_pred = model.get_preds()
            contact_mask = df_pred[:, 0, :] < 0.08
            data_dict['contact_mask_o'] = contact_mask

            J, _, _ = smpl.get_landmarks()
            # obj_center_pred = J[:, 8] + torch.mean(centers_pred[:, 3:, :], -1)
            data_dict["obj_center_pred"] = obj_center_pred
            data_dict["smpl_center_pred"] = data_dict['smpl_center']
            data_dict['smpl_center_act'] = J[:, 8]

            # visualize the neighbours of object points
            # samples = self.get_object_samples(object)
            # model.query(samples, **data_dict['query_dict'])
            # df_pred, pca_pred, parts_pred = model.get_preds()[:3]
            # # mask = df_pred[0, 1] < 0.1
            # mask = df_pred[0, 1] > 0.
            # neighbours = samples[0, mask].detach().cpu().numpy()
            # neigh_o = Mesh(v=neighbours, f=[])
            # neigh_o.set_vertex_colors_from_weights(df_pred[0, 1, mask].detach().cpu().numpy())
            # data_dict['neigh_o'] = neigh_o

            # visualize silhouette
            if image is not None and edges is not None:
                ind = 0
                img = (image[ind].detach().cpu().numpy() * 255).astype(np.uint8)
                img_ref = (data_dict['image_ref'][ind].detach().cpu().numpy() * 255).astype(np.uint8)
                edt = (edges[ind].detach().cpu().numpy() * 255).astype(np.uint8)
                edt_ref = (data_dict['edt_ref'][ind].detach().cpu().numpy() * 255).astype(np.uint8)
                h, w = img.shape[:2]
                vis1, vis2 = np.zeros((h, w, 3)),np.zeros((h, w, 3))
                vis1[:, :, 0] = img_ref # blue: the ref mask
                vis1[:, :, 2] = img
                vis2[:, :, 0] = edt_ref # black: the ref edge
                vis2[:, :, 2] = edt
                comb = np.concatenate([vis1, vis2], 1)
                cv2.imshow('silhouettes', comb)
                cv2.moveWindow('silhouettes', 600, 50)

            self.visualize_fitting(data_dict, object, smpl, smpl_verts)

            cv2.waitKey(10)
