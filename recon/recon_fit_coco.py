"""
joint optimization for coco dataset

if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
sys.path.append(os.getcwd())
import torch

from data.data_paths import DataPaths
from recon.recon_fit_behave import ReconFitterBehave, RECON_PATH
from data.test_data import TestData


class ReconFitterCoco(ReconFitterBehave):
    def init_dataloader(self, args):
        batch_size = args.batch_size
        image_files = DataPaths.get_image_paths_seq(self.seq_folder, check_occlusion=False)
        batch_end = args.end if args.end is not None else len(image_files)
        image_files = image_files[args.start:batch_end]
        dataset = TestData(image_files, batch_size, batch_size,
                           image_size=args.net_img_size,
                           crop_size=args.loadSize,
                           use_mean_center=True) # move human object patch to mean crop center in training dataset
        loader = dataset.get_loader(shuffle=False)
        print(f"In total {len(loader)} batches, {len(image_files)} test examples")
        return loader

    def scale_body_kpts(self, kpts, resize_scale, crop_scale, old_crop_center):
        """
        get the kpts coordinate in cropped image, the crop center is moved to mean crop center
        :param kpts: (B, 25, 3) 2d kpts in original image coordinate before any resizing, crop and scaling
        :param resize_scale: (B, ) image resizing factor to make the image 2048 p
        :param crop_scale: (B, ) cropping scale factor to make the person depth close to 2.2m
        :param old_crop_center: (B, 2), the crop center computed in original image
        :return:
        """
        B = old_crop_center.shape[0]
        crop_center = torch.tensor([[1008.,  995.]]*B).to(self.device) # mean crop center in training set
        # Scale to 1536p image
        pxy = kpts[:, :, :2] * resize_scale.unsqueeze(1).unsqueeze(1)  # coordinate in 2048p image
        # translate with new center
        pxy = pxy - old_crop_center.unsqueeze(1) + crop_center.unsqueeze(1)

        # other operation remains the same
        crop_size_org = crop_scale * self.camera.crop_size  # actual cropping size happened in 2048p image (B,)
        pxy = pxy - crop_center.unsqueeze(1) + crop_size_org.unsqueeze(1).unsqueeze(1) / 2  # coordinate in cropped image
        pxy = pxy * self.net_in_size / crop_size_org.unsqueeze(1).unsqueeze(1)  # coordinate in network input image
        return torch.cat([pxy, kpts[:, :, 2:3]], -1)

    def get_loss_weights(self):
        # stronger pose, contact and 2d keypoints regularization
        loss_weight = {
            'beta':lambda cst, it: 10. ** 0 * cst / (1 + it),
            'pose':lambda cst, it: 10. ** -5 * cst / (1 + it),
            'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'j2d': lambda cst, it: 0.8 ** 2 * cst / (1 + it),
            'object': lambda cst, it: 90.0 ** 2 * cst / (1 + it),
            'part':lambda cst, it: 0.05 ** 2* cst/ (1 + it),
            'contact': lambda cst, it: 150.0 ** 2 * cst / (1 + it),
            'scale': lambda cst, it: 2.0 ** 2 * cst / (1 + it),
            'df_h':lambda cst, it: 30.0 ** 2 * cst / (1 + it),
            'smplz': lambda cst, it: 30 ** 2 * cst / (1 + it),
            'pinit': lambda cst, it: 10 ** 2 * cst / (1 + it),
            'ocent': lambda cst, it: 30 ** 2 * cst / (1 + it),
            'mask': lambda cst, it: 0.3 ** 2 * cst / (1 + it),
            'collide': lambda cst, it: 15 ** 2 * cst / (1 + it),
            'trans': lambda cst, it: 10.0 ** 2 * cst / (1 + it),
        }
        return loss_weight


def recon_fit(args):
    fitter = ReconFitterCoco(args.seq_folder, debug=args.display, obj_name=args.obj_name, args=args)

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
    parser.add_argument('-ck', '--checkpoint', default=None,
                        help='load which checkpoint, will find best or last checkpoint if None')
    parser.add_argument('-fv', '--filter_val', type=float, default=0.004,
                        help='threshold value to filter surface points')
    parser.add_argument('-st', '--sparse_thres', type=float, default=0.03,
                        help="filter value to filter sparse point clouds")
    parser.add_argument('-t', '--tid', default=1, type=int, help='test on images from which kinect')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='optimization batch size')
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-d', '--display', default=False, action='store_true')
    parser.add_argument('-fs', '--start', default=0, type=int, help='start fitting from which frame')
    parser.add_argument('-fe', '--end', default=None, type=int, help='end fitting at which frame')
    parser.add_argument('-on', '--obj_name', default=None,
                        help='object category name, if not provided, will load from sequence information file')

    args = parser.parse_args()

    configs = load_configs(args.exp_name)
    configs.batch_size = args.batch_size
    configs.test_kid = args.tid
    configs.filter_val = args.filter_val
    configs.sparse_thres = args.sparse_thres
    configs.seq_folder = args.seq_folder
    configs.obj_name = args.obj_name

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
