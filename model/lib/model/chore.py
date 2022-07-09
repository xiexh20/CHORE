import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .HGFilters import HGFilter
from ..net_util import init_weights
from .camera import KinectColorCamera


class CHORE(BasePIFuNet):
    def __init__(self,opt,
                 projection_mode='perspective',
                 error_term=nn.MSELoss(),
                 rank=-1,
                 num_parts=14,
                 hidden_dim=128):
        """
        model to predict UDFs, parts, and object poses
        :param opt:
        :param projection_mode:
        :param error_term:
        :param rank:
        :param num_parts:
        :param hidden_dim:
        """
        super(CHORE, self).__init__(projection_mode=projection_mode,
            error_term=error_term)
        self.opt = opt
        self.name = 'chore_v2'

        self.surface_classifier = opt.surface_classifier

        self.image_filter = HGFilter(opt)  # encoder

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []
        # init_net(self, gpu_ids=opt.gpu_ids) # can be set to run in multiple GPUs

        self.bin_classifier = opt.bin_classifier
        self.z_feat = opt.z_feat
        assert self.z_feat in ['zcat', 'zcat2', 'zcat3', 'zcat4', 'zcat5', 'xyz']  # use combined z feature
        print('zfeat: ', self.z_feat)

        zfeat_size = 2
        if self.z_feat == 'zcat3':
            zfeat_size = 25 + 1  # depth on 25 body keypoints
        elif self.z_feat == 'zcat4':
            zfeat_size = 25 + 1  # depth on 25 body keypoints, no perspective normalization
        elif self.z_feat == 'zcat5':
            zfeat_size = 25 + 1 + 1  # object center, real depth, and body keypoints
        elif self.z_feat == 'xyz':
            zfeat_size = 3  # use abs xyz value as feature
        elif self.z_feat == 'realz':
            zfeat_size = 1  # use real depth value as feature

        self.displacement = None
        feat_multi = 1
        if 'displacement' in self.opt and self.opt.displacement:
            print("Using displacements")
            self.displacement = self.make_displacement()
            feat_multi = 5  # 5 times more feature

        self.cls_code = False if 'cls_code' not in self.opt else self.opt.cls_code
        if self.cls_code:
            zfeat_size += 20  # one-hot encoding of object class

        if opt.skip_hourglass:
            feature_size = 256 * feat_multi + zfeat_size + 256 // 4
        else:
            feature_size = 256 * feat_multi + zfeat_size

        self.add_crop_center = False if 'add_crop_center' not in self.opt else self.opt.add_crop_center
        if self.add_crop_center:
            feature_size += 2

        # add smpl center in 2d image, i.e. 8th op joint
        self.add_center_2d = False if 'add_center_2d' not in self.opt else self.opt.add_center_2d
        if self.add_center_2d:
            feature_size += 2  # 2 additional feature for 2d center of body joints

        feature_size = self.additional_feat_size(feature_size)
        self.feature_size = feature_size

        # Human + Object DF predictor
        self.df = self.make_decoder(feature_size, 2, 1, hidden_dim)

        # per-part correspondence predictor
        self.part_predictor = self.make_decoder(feature_size, num_parts, 1, hidden_dim)

        # object pca_axis predictor
        self.pca_predictor = self.make_decoder(feature_size, 9, 1, hidden_dim)

        self.fc_parts_softmax = nn.Softmax(1)

        self.device = torch.device(opt.gpu_id)

        self.camera = KinectColorCamera(opt.loadSize)

        self.point_fc = nn.Conv1d(3, 1, 1)  # one layer to encode point coordinate info
        # loss functions
        self.rank = rank  # for distributed training

        assert not self.surface_classifier
        self.dfloss_func = nn.L1Loss(reduction='none').cuda(self.rank)  # use udf loss

        self.part_loss_func = nn.CrossEntropyLoss(reduction='none').cuda(self.rank)
        self.class_loss_func = nn.CrossEntropyLoss(reduction='none').cuda(self.rank)

        self.OUT_DIST = 5.0  # value for points outside the image plane
        self.realdepth = opt.realdepth
        if self.realdepth:
            print("z_feat on real depth")

        init_weights(self)
        self.error_buffer = None  # buffer to save errors

        # buffer for error computation
        self.points = None
        self.offsets = None
        self.crop_center = None
        self.obj_center = None

        # for gradient computation
        self.point_local_feats = []

        # default loss weights
        self.loss_weights = [1.0, 1.0, 0.006, 500, 1000, 1000, 0.02,
                             0.1]  # dfh, dfo, parts, pca,  smpl, obj, gradh grado

        self.noclip = False if 'noclip' not in self.opt else self.opt.noclip

        # add smpl center and obj center decoder, order matters!
        self.center_predictor = self.make_decoder(self.feature_size,
                                                  6, 1, hidden_dim)

    def additional_feat_size(self, feat_size):
        return feat_size

    def make_displacement(self):
        displacment = 0.0722
        displacments = [[0, 0]]
        for x in range(2):
            for y in [-1, 1]:
                input = [0, 0]
                input[x] = y * displacment
                displacments.append(input)

        return torch.nn.Parameter(torch.tensor(displacments), requires_grad=False)

    def make_decoder(self, input_sz, output_sz, group_sz, hidden_sz):
        # per-part occupancy predictor
        predictor = [
            nn.Conv1d(input_sz, hidden_sz * group_sz, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_sz * group_sz, hidden_sz * group_sz, 1, groups=group_sz),
            nn.ReLU(),
            nn.Conv1d(hidden_sz * group_sz, hidden_sz * group_sz, 1, groups=group_sz),
            nn.ReLU(),
            nn.Conv1d(hidden_sz * group_sz, output_sz, 1, groups=group_sz)
        ]
        return nn.Sequential(*predictor)

    def filter(self, images):
        '''
        Filter the input images, normalized or not?
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def project_points(self, points, offsets):
        """
        points: (B, N, 3) points in local camera coordinate system
        offsets: coordinate offset due to cropping (B, 2)
        return: (B, 3, N) for compatibility

        """
        # adapt for multiple GPU training
        # cam = PerspectiveCameras(focal_length=-self.focal, principal_point=self.center,
        #                          image_size=(self.cam_imgsize,),
        #                          device=points.device)
        # xyz = cam.transform_points(points)
        # return self.camera(points, offsets)
        return self.camera.project_points(points, offsets)

    def query(self, points, offsets=None, transforms=None, labels=None, body_kpts=None,
              obj_center=None, crop_center=None, **kwargs):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, N, 3] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        # add data to buffer
        self.points = points
        self.offsets = offsets
        self.crop_center = crop_center  # (B, 2)

        xy, z_feat = self.get_zfeat(body_kpts, crop_center, obj_center, offsets, points)
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        addi_feat = self.get_additional_feat(points)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        point_local_feats = []

        for im_feat in self.im_feat_list:
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            if self.opt.skip_hourglass:  # use skip connection? yes!
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)  # dimension of feature size?

            point_local_feats.append(point_local_feat)  # for gradient computation

            preds = self.decode(point_local_feat)

            # out of image plane is always set to a maximum
            df = preds[0]  # the first is always df prediction
            df_trans = df.transpose(1, 2)  # (B, 2, N) -> (B, N, 2)
            df_trans[~in_img] = self.OUT_DIST
            df = df_trans.transpose(1, 2)

            self.intermediate_preds_list.append((df, *preds[1:]))

        self.point_local_feats = point_local_feats
        self.preds = self.intermediate_preds_list[-1]

    def get_additional_feat(self, points):
        return None

    def get_zfeat(self, body_kpts, crop_center, obj_center, offsets, points):
        "get the zfeat for query points"
        xyz = self.project_points(points, crop_center)
        xy = xyz[:, :2, :]  # xyz are transposed to (B, 3, N)
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        # print('{} out of {} in image'.format(torch.sum(in_img), xyz.shape[0] * xyz.shape[2]))
        if self.z_feat == 'vector':
            z_feat = (points - offsets.unsqueeze(1)).transpose(1, 2)
        elif self.z_feat == 'zonly':
            z_feat = (points[:, :, 2:3] - offsets[:, 2:3].unsqueeze(1)).transpose(1, 2)
        elif self.z_feat == 'zinv':
            z_feat = self.normalizer(z)  # old version
            p_feat = self.point_fc(torch.transpose(points, 2, 1))
            z_feat = torch.cat([z_feat, p_feat], 1)
        elif self.z_feat == 'zperspective':
            z_center = offsets[:, 2:3].unsqueeze(1)  # TODO: on dataloader, make sure z_center is not zero
            z_feat = ((points[:, :, 2:3] - z_center) / z_center).transpose(1, 2)
        elif self.z_feat == 'zcat':
            z_center = offsets[:, 2:3].unsqueeze(1)  # (B, 1, 1)
            z_feat1 = ((points[:, :, 2:3] - z_center) / z_center).transpose(1, 2)  # perspective depth feature
            # z_feat2 = (points[:, :, 2:3] - 2.5).transpose(1, 2)  # real depth, no relative
            z_feat2 = self.get_zfeat2(points)
            z_feat = torch.cat([z_feat1, z_feat2], 1)
        elif self.z_feat == 'zcat2':
            z_center = offsets[:, 2:3].unsqueeze(1)  # (B, 1, 1)
            z_feat1 = ((points[:, :, 2:3] - z_center) / points[:, :, 2:3]).transpose(1, 2)  # perspective depth feature
            # z_feat2 = (points[:, :, 2:3] - 2.5).transpose(1, 2)  # real depth, no relative
            z_feat2 = self.get_zfeat2(points)
            z_feat = torch.cat([z_feat1, z_feat2], 1)
        elif self.z_feat == 'zcat3':
            kpts_depths = body_kpts[:, :, 2:3].transpose(1, 2)  # (B, 25, 1) -> (B, 1, 25)
            # use 25 body keypoints as features
            z_feat1 = ((points[:, :, 2:3] - kpts_depths) / kpts_depths).transpose(1, 2)  # (B, 25, N)
            z_feat2 = self.get_zfeat2(points)
            z_feat = torch.cat([z_feat1, z_feat2], 1)
        elif self.z_feat == 'zcat4':
            # use 25 body keypoints as features
            kpts_depths = body_kpts[:, :, 2:3].transpose(1, 2)  # (B, 25, 1) -> (B, 1, 25)
            # no normalization, just relative depth
            z_feat1 = (points[:, :, 2:3] - kpts_depths).transpose(1, 2)  # (B, 25, N)
            z_feat2 = self.get_zfeat2(points)
            z_feat = torch.cat([z_feat1, z_feat2], 1)
        elif self.z_feat == 'zcat5':
            # 25 body keypoints + object center + real depth
            kpts_depths = body_kpts[:, :, 2:3].transpose(1, 2)  # (B, 25, 1) -> (B, 1, 25)
            # no normalization, just relative depth
            z_feat1 = (points[:, :, 2:3] - kpts_depths).transpose(1, 2)  # (B, 25, N)
            z_feat2 = self.get_zfeat2(points)
            z_obj_center = obj_center[:, 2:3].unsqueeze(1)  # (B, 1, 1)
            z_feat3 = ((points[:, :, 2:3] - z_obj_center) / z_obj_center).transpose(1, 2)  # perspective depth feature
            z_feat = torch.cat([z_feat1, z_feat2, z_feat3], 1)
        elif self.z_feat == 'realz':
            # use real depth, no normalization
            z_feat = (points[:, :, 2:3]).transpose(1, 2)
        elif self.z_feat == 'xyz' and self.opt.projection_mode == 'perspective':
            # print("xyz feature")
            # print(points.dtype)
            rela_z = (points[:, :, 2:3] - 2.2).transpose(1, 2)  # relative depth to a fixed smpl center
            z_feat = torch.cat([points[:, :, 0:2].transpose(1, 2), rela_z], 1)  # use xyz values
            # print("xyz feature")
        elif self.z_feat == 'xyz' and self.opt.projection_mode == 'orthographic':
            # print("orth xyz feature")
            z_feat = points.transpose(1, 2)  # use xyz values
        return xy, z_feat

    def get_zfeat2(self, points):
        if self.realdepth:
            z_feat2 = (points[:, :, 2:3] - 2.5).transpose(1, 2)  # for backward compatability
        else:
            z_feat2 = (points[:, :, 2:3]).transpose(1, 2)  # real depth, no relative
        return z_feat2

    def decode(self, features):
        "predict pca, smpl and object center"
        df = self.df(features)
        pca_axis = self.pca_predictor(features)
        out_pca = pca_axis.view(df.shape[0], 3, 3, -1)
        parts = self.part_predictor(features)

        centers = self.center_predictor(features)

        if self.opt.joint_df:
            part_softmax = self.fc_parts_softmax(parts)
            df_comb = df * part_softmax
            df_out = df_comb.mean(1)
        else:
            df_out = df

        return df_out, out_pca, parts, centers
        # if self.bin_classifier:
        #     df, parts, class_labels = self.decoder(features)
        #     return df, parts, class_labels
        # else:
        #     df, parts = self.decoder(features)
        #     return df, parts

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            df, labels = preds
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)

        return error

    def get_separate_losses(self):
        return self.error_buffer

    def forward(self, images, points, df_h, df_o, parts_gt, pca_gt, offsets=None,
                max_dist=5.0, body_kpts=None, obj_center=None, crop_center=None,
                **kwargs):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, offsets=offsets, transforms=None,
                   labels=None, body_kpts=body_kpts,
                   obj_center=obj_center, crop_center=crop_center,
                   **kwargs)

        # get the prediction
        # res = self.get_preds()  # this get the last layer out

        # predict centers as well
        error = self.get_errors(df_h, df_o, parts_gt, pca_gt, max_dist,
                                offsets, obj_center, **kwargs)

        return error

    def get_errors(self, df_h, df_o, parts_gt, pca_gt, max_dist, body_center,
                   obj_center, **kwargs):
        """
        body_center: smpl center
        object center: object center relative to smpl center
        """

        losses_all, error = 0.0, 0.
        for preds in self.intermediate_preds_list:
            df_pred, pca_pred, parts_pred, centers = preds
            if self.opt.joint_df:
                df_joint = df_h
                obj_mask = df_o < df_h
                df_joint[obj_mask] = df_o[obj_mask]
                loss_df = self.get_df_loss(df_pred, df_joint, max_dist)
                loss_h = loss_df * 2.5
                loss_o = 0.
            else:
                # separate distance field to human and object
                df_h_pred = df_pred[:, 0]  # (B, N)
                df_o_pred = df_pred[:, 1]
                loss_h = self.get_df_loss(df_h, df_h_pred, max_dist) * self.loss_weights[0]
                loss_o = self.get_df_loss(df_o, df_o_pred, max_dist) * self.loss_weights[1]

            # loss_parts = self.part_loss_func(parts_pred, parts_gt) * 0.1
            loss_parts = self.part_loss_func(parts_pred, parts_gt) * self.loss_weights[2]
            loss_parts = loss_parts.sum(-1).mean()

            # PCA axis loss
            mask = (df_o < 0.05).unsqueeze(1).unsqueeze(1)  # (B, N), pca_gt: (B, 3, 3, N)
            # print('pca_pred shape: {}, gt shape: {}'.format(pca_pred.shape, pca_gt.shape))
            # loss_pca = (F.mse_loss(pca_pred, pca_gt, reduction='none') * mask) * 10. ** 3
            loss_pca = (F.mse_loss(pca_pred, pca_gt, reduction='none') * mask) * self.loss_weights[3]
            loss_pca = loss_pca.mean()

            # object center  prediction loss
            loss_obj_center = (F.mse_loss(centers[:, 3:, :], obj_center, reduction='none') * mask)
            loss_obj_center = loss_obj_center.mean() * self.loss_weights[4]

            # smpl center prediction loss
            mask = (df_h < 0.05).unsqueeze(1)  # (B, N) -> (B, 1, N)
            # print("mask: {}, centers: {}".format(mask.shape, body_center.shape))
            B, _, N = mask.shape[:3]
            loss_smpl_center = (F.mse_loss(centers[:, :3, :], body_center.unsqueeze(-1).repeat(1, 1, N),
                                           reduction='none') * mask)
            loss_smpl_center = loss_smpl_center.mean() * self.loss_weights[5]

            error += loss_h + loss_o + loss_parts + loss_pca + loss_smpl_center + loss_obj_center
            losses_all += torch.tensor([loss_h, loss_o, loss_parts, loss_pca, loss_smpl_center, loss_obj_center])

        error /= len(self.intermediate_preds_list)
        losses_all /= len(self.intermediate_preds_list)

        self.error_buffer = losses_all
        self.print_errors(losses_all)

        return error, losses_all

    def get_grad_loss(self, grad_pred, grad_gt, df_pred, max_dist):
        "grad: (B, N, 3)"
        # nan_mask = torch.isnan(torch.sum(grad_gt, -1))
        valid = df_pred < max_dist
        err = (1.0 - F.cosine_similarity(grad_pred, grad_gt, dim=-1)) * valid
        # err[torch.isnan(err)] = 0.  # TODO: remove invalid gradients when generating data
        err = err.sum(-1).mean()
        return err

    def get_df_loss(self, df_gt, df_pred, max_dist):
        loss_df = self.dfloss_func(torch.clamp(df_pred, max=max_dist),
                                   torch.clamp(df_gt, max=max_dist))
        return loss_df.sum(-1).mean()

    def format_sep_losses(self, losses_all):
        "losses_all: loss tensor"
        names = ['df_h', 'df_o', 'parts', 'pca', 'smpl', 'obj']
        loss_dict = {}
        for name, sp_loss in zip(names, losses_all):
            loss_dict[name] = sp_loss
        return loss_dict

    def print_errors(self, errors):
        names = ['df_h', 'df_o', 'parts', 'pca', 'smpl', 'obj', 'grad_h', 'grad_o']
        lstr = ''
        for n, ls in zip(names, errors):
            lstr += f'{n}:{ls}, '
        print(lstr[:-1])



