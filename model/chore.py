import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .HGFilters import HGFilter
from model.net_util import init_weights
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
        self.name = 'chore'
        self.device = torch.device(opt.gpu_id)

        self.image_filter = HGFilter(opt)  # encoder

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.intermediate_preds_list = []

        self.z_feat = opt.z_feat
        assert self.z_feat in ['zcat', 'zcat2', 'zcat3', 'zcat4', 'zcat5', 'xyz']  # use combined z feature

        assert self.z_feat == 'xyz'
        zfeat_size = 3
        feature_size = 256 + zfeat_size + 256 // 4

        # Decoders
        # Human + Object DF predictor
        self.df = self.make_decoder(feature_size, 2, 1, hidden_dim)
        # per-part correspondence predictor
        self.part_predictor = self.make_decoder(feature_size, num_parts, 1, hidden_dim)
        # object pca_axis predictor
        self.pca_predictor = self.make_decoder(feature_size, 9, 1, hidden_dim)
        # smpl center and obj center decoder
        self.center_predictor = self.make_decoder(feature_size, 6, 1, hidden_dim)

        # loss functions
        self.rank = rank  # for distributed training
        self.dfloss_func = nn.L1Loss(reduction='none').cuda(self.rank)  # use udf loss
        self.part_loss_func = nn.CrossEntropyLoss(reduction='none').cuda(self.rank)
        # default loss weights for dfh, dfo, parts, pca,  smpl, obj, gradh grado
        self.loss_weights = [1.0, 1.0, 0.006, 500, 1000, 1000]

        self.camera = KinectColorCamera(opt.loadSize)
        self.OUT_DIST = 5.0  # value for points outside the image plane

        init_weights(self)

        # buffer for error computation
        self.points = None
        self.crop_center = None
        self.error_buffer = None

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
        return: (B, 3, N) projected points, after normalization
        """
        # adapt for multiple GPU training
        return self.camera.project_points(points, offsets)

    def query(self, points, crop_center=None, **kwargs):
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
        # self.offsets = offsets
        self.crop_center = crop_center  # (B, 2)

        # xy, z_feat = self.get_zfeat(body_kpts, crop_center, obj_center, offsets, points)
        xyz = self.project_points(points, crop_center)
        xy = xyz[:, :2, :]  # xyz are transposed to (B, 3, N)
        assert self.z_feat == 'xyz' and self.opt.projection_mode == 'perspective'
        rela_z = (points[:, :, 2:3] - 2.2).transpose(1, 2)  # relative depth to a fixed smpl center
        z_feat = torch.cat([points[:, :, 0:2].transpose(1, 2), rela_z], 1)  # use xyz values
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        assert self.opt.skip_hourglass
        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            if self.opt.skip_hourglass:  # use skip connection? yes!
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)
            preds = self.decode(point_local_feat)

            # out of image plane is always set to a maximum
            df = preds[0]  # the first is always df prediction
            df_trans = df.transpose(1, 2)  # (B, 2, N) -> (B, N, 2)
            df_trans[~in_img] = self.OUT_DIST
            df = df_trans.transpose(1, 2)

            self.intermediate_preds_list.append((df, *preds[1:]))

        self.preds = self.intermediate_preds_list[-1]

    def decode(self, features):
        "predict pca, smpl and object center"
        df = self.df(features)
        pca_axis = self.pca_predictor(features)
        out_pca = pca_axis.view(df.shape[0], 3, 3, -1)
        parts = self.part_predictor(features)

        centers = self.center_predictor(features)

        df_out = df

        return df_out, out_pca, parts, centers

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def forward(self, images, points, df_h, df_o, parts_gt, pca_gt, body_center=None,
                max_dist=5.0, obj_center=None, crop_center=None,
                **kwargs):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, crop_center=crop_center,
                   **kwargs)

        # predict centers as well
        error = self.get_errors(df_h, df_o, parts_gt, pca_gt, max_dist,
                                body_center, obj_center, **kwargs)

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
            # separate distance fields to human and object
            df_h_pred = df_pred[:, 0]  # (B, N)
            df_o_pred = df_pred[:, 1]
            loss_h = self.get_df_loss(df_h, df_h_pred, max_dist) * self.loss_weights[0]
            loss_o = self.get_df_loss(df_o, df_o_pred, max_dist) * self.loss_weights[1]

            # loss_parts = self.part_loss_func(parts_pred, parts_gt) * 0.1
            loss_parts = self.part_loss_func(parts_pred, parts_gt) * self.loss_weights[2]
            loss_parts = loss_parts.sum(-1).mean()

            # PCA axis loss
            mask = (df_o < 0.05).unsqueeze(1).unsqueeze(1)  # (B, N), pca_gt: (B, 3, 3, N)
            loss_pca = (F.mse_loss(pca_pred, pca_gt, reduction='none') * mask) * self.loss_weights[3]
            loss_pca = loss_pca.mean()

            # object center  prediction loss
            loss_obj_center = (F.mse_loss(centers[:, 3:, :], obj_center, reduction='none') * mask)
            loss_obj_center = loss_obj_center.mean() * self.loss_weights[4]

            # smpl center prediction loss
            mask = (df_h < 0.05).unsqueeze(1)  # (B, N) -> (B, 1, N)
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



