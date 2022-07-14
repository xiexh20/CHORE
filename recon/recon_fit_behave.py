"""
reconstruct human+object on behave dataset

if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
sys.path.append(os.getcwd())
from tqdm import tqdm
from os.path import basename
import torch
import torch.optim as optim
import torch.nn.functional as F

from lib_smpl.const import SMPL_POSE_PRAMS_NUM

from data.data_paths import DataPaths
from data.test_data import TestData
from model import CHORE
from recon.generator import Generator
from recon.obj_pose_roi import SilLossROI
from recon.recon_fit_base import ReconFitterBase, RECON_PATH


class ReconFitterBehave(ReconFitterBase):
    def fit_recon(self, args):
        # prepare dataloader
        loader = self.init_dataloader(args)
        # prepare model
        model = CHORE(args)
        # generator
        generator = Generator(model, args.exp_name, threshold=2.0,
                              sparse_thres=args.sparse_thres,
                              filter_val=args.filter_val)
        loop = tqdm(loader)
        loop.set_description(basename(args.seq_folder))

        for i, data in enumerate(loop):
            batch_size = data['images'].shape[0]
            if self.is_done(data['path'], args.save_name, args.test_kid) and not args.redo:
                print(data['path'], args.save_name, 'already done, skipped')
                continue
            pc_generated = generator.generate_pclouds_batch(data, num_points=5000, num_steps=10, mute=True)

            # obtain SMPL init
            betas_dict, body_kpts, human_parts, human_points, human_t, obj_points, part_colors, part_labels, query_dict, smpl = self.prep_smplfit(
                data, generator, pc_generated)

            smpl, scale = self.optimize_smpl(smpl, betas_dict, iter_for_kpts=1, iter_for_pose=1, iter_for_betas=1) # coco data

            # obtain object init
            obj_R, obj_s, obj_t, object_init = self.init_obj_fit_data(batch_size, human_t, pc_generated, scale)

            data_dict = {
                'obj_R': obj_R,
                'obj_t': obj_t,
                'obj_s': obj_s,
                'objects': object_init,
                'smpl': smpl,
                'images': data.get('images').to(self.device),
                'human_init': human_points,
                'obj_init': obj_points,
                'human_parts': human_parts,
                'part_labels': part_labels,
                'part_colors': part_colors,
                'body_kpts': body_kpts,
                'query_dict': query_dict,
                'obj_t_init': obj_t.clone().detach().to(self.device)
            }

            smpl, obj_R, obj_t = self.optimize_smpl_object(generator.model, data_dict)

            self.save_outputs(smpl, obj_R, obj_t, data['path'], args.save_name, args.test_kid, obj_s)

    def init_dataloader(self, args):
        batch_size = args.batch_size
        image_files = DataPaths.get_image_paths_seq(self.seq_folder, check_occlusion=True)
        batch_end = args.end if args.end is not None else len(image_files)
        image_files = image_files[args.start:batch_end]
        dataset = TestData(image_files, batch_size, batch_size,
                           image_size=args.net_img_size,
                           crop_size=args.loadSize)
        loader = dataset.get_loader(shuffle=False)
        print(f"In total {len(loader)} test examples")
        return loader

    def optimize_smpl_object(self, model, data_dict, obj_iter=20, joint_iter=10, steps_per_iter=10):
        """
        use silhouette loss to fine tune rotation only, optimize until convergence
        """
        images = data_dict['images']
        sil = SilLossROI(images[:, 3, :, :], images[:, 4, :, :], self.scan, data_dict['query_dict']['crop_center'])
        data_dict['silhouette'] = sil.to(self.device)

        smpl = data_dict['smpl']
        smpl_split = self.split_smpl(smpl)
        data_dict['smpl'] = smpl_split
        obj_R, obj_t, obj_s = data_dict['obj_R'], data_dict['obj_t'], data_dict['obj_s']
        obj_optimizer = optim.Adam([obj_t, obj_R, obj_s], lr=0.006)

        weight_dict = self.get_loss_weights()
        iter_for_global, iter_for_separate, iter_for_smpl_pose = 0, 0, 0
        iter_for_obj, iterations = obj_iter, joint_iter
        iter_for_sil = 50 # optimize rotation only
        max_iter, prev_loss = 100, 300.

        # iter_for_obj, max_iter = 1, 20 # for debug only

        # now compute smpl center once
        data_dict['smpl_center'] = self.compute_smpl_center_pred(data_dict, model, smpl)

        loop = tqdm(range(iterations + iter_for_obj + max_iter + iter_for_sil))

        for it in loop:
            obj_optimizer.zero_grad()

            description = ''
            if it < iter_for_obj:
                description = 'optimizing object only'
                phase = 'object only'
            elif it == iter_for_obj:
                phase = 'sil'
                obj_optimizer = optim.Adam([obj_R, obj_s, obj_t], lr=0.006) # optimize rotation only using silhouette loss
                rot_init = self.decopose_axis(data_dict['obj_R']).detach().clone()
                data_dict['rot_init'] = rot_init
                data_dict['trans_init'] = data_dict['obj_t'].detach().clone()
                description = 'optimizing with silhouette'
            elif it == iter_for_obj + iter_for_sil:
                description = 'joint optimization'
                phase = 'joint'
                obj_optimizer = optim.Adam([obj_t, obj_s], lr=0.002)  # smaller learning rate, optimize translation only

            for i in range(steps_per_iter):
                loss_dict = self.forward_step(model, smpl_split, data_dict, obj_R, obj_t, obj_s, phase)

                if loss_dict == 0.:
                    print('early stopped at iter {}'.format(it))
                    return smpl, data_dict['obj_R'], data_dict['obj_t']

                weight_decay = 1 if phase == 'object only' else it
                if phase == 'sil':
                    weight_decay = it - iter_for_obj + 1
                elif phase == 'joint':
                    weight_decay = (it - (iter_for_global + iter_for_smpl_pose + iter_for_obj) + 1) / 5

                loss = self.sum_dict(loss_dict, weight_dict, weight_decay)

                loss.backward()
                obj_optimizer.step()

                l_str = 'Iter: {} decay: {}'.format(f"{it}-{i}", weight_decay)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], weight_decay).mean().item())
                loop.set_description(f"{description} {l_str}")
                if (abs(prev_loss - loss) / prev_loss < prev_loss * 0.0001) and (
                        it > 0.25 * max_iter) and phase == 'joint':
                    return smpl, data_dict['obj_R'], data_dict['obj_t']
                prev_loss = loss

        return smpl, data_dict['obj_R'], data_dict['obj_t']

    def forward_step(self, model, smpl, data_dict, obj_R, obj_t, obj_s, phase):
        """
        one forward step for joint optimization
        """
        smpl_verts, _, _, _ = smpl()
        loss_dict = {}

        # object losses
        object_init = data_dict['objects']
        rot = obj_R
        R = self.decopose_axis(rot)

        # evaluate object DF at object points
        object = self.transform_obj_verts(object_init, R, obj_t, obj_s)
        model.query(object, **data_dict['query_dict'])
        preds = model.get_preds()
        df_pred, _, _, centers_pred_o = preds
        part_o = preds[2]  # for contact loss
        obj_center_pred = data_dict['smpl_center'] + torch.mean(centers_pred_o[:, 3:, :], -1)

        image, edges = None, None
        if phase == 'sil':
            # using only silhouette loss to optimize rotation
            sil = data_dict['silhouette']  # silhouette loss
            obj_losses, image, edges, image_ref, edt_ref = sil(R, obj_t, obj_s)
            loss_dict['mask'] = obj_losses['mask']
            data_dict['image_ref'] = image_ref
            data_dict['edt_ref'] = edt_ref # for visualization
            loss_dict['scale'] = torch.mean((obj_s - self.obj_scale) ** 2) # 2D losses are prone to local minimum, need regularization
            loss_dict['trans'] = torch.mean((obj_t - data_dict['trans_init']) ** 2)
        else:
            preds = self.compute_obj_loss(data_dict, loss_dict, model, obj_s, object)
            # use ocent to regularize for object only optimization
            obj_center_act = torch.mean(object, 1)
            loss_dict['ocent'] = F.mse_loss(obj_center_act, obj_center_pred, reduction='none').sum(-1).mean()

            if phase == 'joint':
                # contact loss
                df_obj_h = df_pred[:, 0, :]
                model.query(smpl_verts, **data_dict['query_dict'])
                preds = model.get_preds()
                df_pred, centers_pred_h = preds[0], preds[-1]

                df_hum_o = df_pred[:, 1, :]

                # comment this for no contact baseline
                self.compute_contact_loss(df_hum_o, df_obj_h, object, smpl_verts, loss_dict, part_o=part_o)

                # prevent interpenetration
                pen_loss = self.compute_collision_loss(smpl_verts, smpl.faces,
                                                       R, obj_t, obj_s)
                loss_dict['collide'] = pen_loss

        if self.debug:
            # visualize
            self.visualize_contact_fitting(data_dict, edges, image, model, obj_center_pred, object, smpl, smpl_verts)

        return loss_dict

    def optimize_smpl(self, smpl, data_dict, iter_for_betas=10, iter_for_pose=10,
                      iter_for_kpts=5, steps_per_iter=10, max_iter=150):
        """
        optimize SMPL until convergence
        first optimize global translation and betas, then optimize all poses,
        finally fine tune with keypoint losses until convergence
        """
        smpl_split = self.split_smpl(smpl)
        smpl_optimizer = optim.Adam([smpl_split.top_betas, smpl_split.trans], lr=0.02)
        height_init = self.get_smpl_height(smpl)

        weight_dict = self.get_loss_weights()
        max_iter, prev_loss = max_iter, 300.

        # max_iter, prev_loss = 1, 300. # for debug
        # iter_for_betas, iter_for_kpts, iter_for_pose = 1, 1, 1
        loop = tqdm(range(iter_for_betas + iter_for_kpts + iter_for_pose + max_iter))

        for it in loop:
            smpl_optimizer.zero_grad()

            loop_inner = range(steps_per_iter)

            if it < iter_for_betas:
                description = 'optimizing smpl beta only'
                phase = 'global'
            elif it == iter_for_betas:
                description = 'optimizing all smpl pose '
                phase = 'smpl all pose'
                smpl_optimizer = torch.optim.Adam([smpl_split.trans,
                                                   smpl_split.global_pose,
                                                   smpl_split.body_pose,
                                                   smpl_split.top_betas,
                                                   smpl_split.other_betas
                                                   # ], 0.002, betas=(0.9, 0.999))
                                                   ], 0.006, betas=(0.9, 0.999))
            elif it < iter_for_betas + iter_for_pose:
                description = 'optimizing all smpl pose '
            elif it == iter_for_betas + iter_for_pose:
                description = 'fine tune with kpts'
                phase = 'kpts'

            for i in loop_inner:
                loss_dict = self.forward_smpl(smpl_split, data_dict, phase)
                decay = 1 if phase != 'kpts' else it / 3
                # decay = 1
                loss = self.sum_dict(loss_dict, weight_dict, decay)

                loss.backward()
                smpl_optimizer.step()

                lstr = self.get_loss_str(f"{it}-{i}", loss_dict, weight_dict, decay)
                loop.set_description(f'{description}: {lstr}')

                if (abs(prev_loss - loss) / prev_loss < prev_loss * 0.001) and (it > 0.25 * max_iter + iter_for_betas + iter_for_pose):
                    print('early stop at,', it)
                    # compute a scale after the opt
                    height_after = self.get_smpl_height(smpl_split)
                    scale = height_after / height_init
                    smpl = self.copy_smpl_params(smpl_split, smpl)
                    return smpl, scale
                prev_loss = loss

        # compute a scale after the opt
        height_after = self.get_smpl_height(smpl_split)
        scale = height_after / height_init
        smpl = self.copy_smpl_params(smpl_split, smpl)
        return smpl, scale

    def forward_smpl(self, smpl, data_dict, phase):
        """
        one forward step for smpl optimization
        :param smpl: batch SMPL instance
        :param data_dict:
        :param phase:
        :return:
        """
        loss_dict = {}
        smpl_part_labels = data_dict['part_labels']
        model = data_dict['net']

        smpl_verts, _, _, _ = smpl()
        df_pred, parts_pred, centers_pred = self.compute_df_h_loss(data_dict, loss_dict, model, smpl_verts)

        # priors
        self.compute_prior_loss(loss_dict, smpl, nobeta=True)

        # part correspondence loss
        loss_dict['part'] = F.cross_entropy(parts_pred, smpl_part_labels, reduction='none').sum(-1).mean()

        # fixed smpl depth
        J, face, hands = smpl.get_landmarks()
        self.smplz_loss(J, loss_dict)

        # mocap init pose
        pose_init = data_dict['pose_init']
        loss_dict['pinit'] = torch.mean(torch.sum((smpl.pose[:, 3:SMPL_POSE_PRAMS_NUM] - pose_init) ** 2, -1))

        if phase == 'kpts':
            # kpts loss
            self.compute_kpts_loss(data_dict, loss_dict, smpl)

        if self.debug:
            with torch.no_grad():
                contact_mask = df_pred[:, 1, :] < 0.08
                data_dict['contact_mask'] = contact_mask

                smpl_center_pred = torch.mean(centers_pred[:, :3, :], -1)
                smpl_center_pred[:, 2] = self.z_0 # fixed depth
                data_dict['smpl_center_pred'] = smpl_center_pred

                self.visualize_smpl_fit(data_dict, smpl, smpl_verts)

        return loss_dict

    def get_loss_weights(self):
        loss_weight = {
            'beta':lambda cst, it: 10. ** 0 * cst / (1 + it), # priors
            'pose':lambda cst, it: 10. ** -5 * cst / (1 + it),
            'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'j2d': lambda cst, it: 0.3 ** 2 * cst / (1 + it), # 2D body keypoints
            'object':lambda cst, it: 30.0 ** 2 * cst / (1 + it), # for mean df_o
            'part': lambda cst, it: 0.05 ** 2 * cst / (1 + it), # for cross entropy loss
            'contact': lambda cst, it: 30.0 ** 2 * cst / (1 + it), # contact loss
            'scale': lambda cst, it: 10.0 ** 2 * cst / (1 + it),  # no scale
            'df_h':lambda cst, it: 30.0 ** 2 * cst / (1 + it), # human distance loss
            'smplz': lambda cst, it: 30 ** 2 * cst / (1 + it), # fixed SMPL depth
            'mask': lambda cst, it: 0.003 ** 2 * cst / (1 + it), # 2D object mask loss
            'ocent': lambda cst, it: 15 ** 2 * cst / (1 + it),  # object center
            'collide': lambda cst, it: 3 ** 2 * cst / (1 + it), # human object collision
            'pinit': lambda cst, it: 5 ** 2 * cst / (1 + it), # initial pose
            'rot':lambda cst, it: 10.0 ** 2 * cst / (1 + it), # prevent deviate too much
            'trans': lambda cst, it: 10.0 ** 2 * cst / (1 + it),  # prevent deviate too much
        }
        return loss_weight


def recon_fit(args):
    fitter = ReconFitterBehave(args.seq_folder, debug=args.display, outpath=args.outpath, args=args)

    fitter.fit_recon(args)
    print('all done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback
    from config.config_loader import load_configs

    parser = ArgumentParser()
    parser.add_argument('exp_name', help='experiment name')
    parser.add_argument('-s', '--seq_folder', help="path to one BEHAVE sequence")
    parser.add_argument('-sn', '--save_name', required=True)
    parser.add_argument('-o', '--outpath', default=RECON_PATH, help='where to save reconstruction results')
    parser.add_argument('-ck', '--checkpoint', default=None, help='load which checkpoint, will find best or last checkpoint if None')
    parser.add_argument('-fv', '--filter_val', type=float, default=0.004, help='threshold value to filter surface points')
    parser.add_argument('-st', '--sparse_thres', type=float, default=0.03, help="filter value to filter sparse point clouds")
    parser.add_argument('-t', '--tid', default=1, type=int, help='test on images from which kinect')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='optimization batch size')
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-d', '--display', default=False, action='store_true')
    parser.add_argument('-fs', '--start', default=0, type=int, help='start fitting from which frame')
    parser.add_argument('-fe', '--end', default=None, type=int, help='end fitting at which frame')

    args = parser.parse_args()

    configs = load_configs(args.exp_name)
    configs.batch_size = args.batch_size
    configs.test_kid = args.tid
    configs.filter_val = args.filter_val
    configs.sparse_thres = args.sparse_thres
    configs.seq_folder = args.seq_folder

    configs.save_name = args.save_name

    configs.checkpoint = args.checkpoint
    configs.outpath = args.outpath

    configs.redo = args.redo
    configs.display = args.display
    configs.start = args.start
    configs.end = args.end

    try:
        recon_fit(configs)
    except:
        log = traceback.format_exc()
        print(log)