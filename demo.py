"""
simple demo to run CHORE on one image

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import os.path as osp
from glob import glob
import pickle as pkl
import numpy as np

import cv2

from recon.recon_fit_coco import ReconFitterCoco
from utils.render_utils import NrWrapper
import utils.render_utils as rutils


def main(args):
    fitter = ReconFitterCoco(args.seq_folder, obj_name=args.obj_name, outpath='./', args=args)
    fitter.fit_recon(args)

    # render results
    H, W = 1536, 2048
    nrwrapper = NrWrapper(image_size=W)
    side_renderer = rutils.setup_side_renderer(2.0, 0., 90.)
    image_folders = sorted(glob(args.seq_folder+"/*/"))
    for folder in image_folders:
        rgb = cv2.imread(folder + "k1.color.jpg")[:, :, ::-1]
        oh, ow = rgb.shape[:2]
        crop_info = pkl.load(open(folder+"k1.crop_info.pkl", 'rb'))
        rgb_1536p = cv2.resize(rgb, crop_info['rgb_newsize'])

        smpl = rutils.load_mesh(folder + f'{args.save_name}/k1.smpl.ply')
        obj = rutils.load_mesh(folder + f'{args.save_name}/k1.object.ply')

        rend, mask = nrwrapper.render_meshes(nrwrapper.front_renderer, [smpl, obj])
        rend = (rend * 255).astype(np.uint8)
        mask = (mask*255).astype(np.uint8)

        # align with input images
        rend_overlap = rutils.align_to_input(crop_info, H, rend, args.loadSize, W, True)
        mask_overlap = rutils.align_to_input(crop_info, H, mask, args.loadSize, W, True, 0)

        mask_in = mask_overlap > 127
        overlap = rgb_1536p.copy()
        overlap[mask_in] = rend_overlap[mask_in]
        cv2.imwrite(folder + f'{args.save_name}/k1.rend_overlap.jpg', cv2.resize(overlap[:, :, ::-1], (ow, oh)))

        # render side view
        faces, texts, verts = nrwrapper.prepare_side_rend([smpl, obj], maxd=1.8)
        rend, mask = nrwrapper.render(side_renderer, verts, faces, texts)
        cv2.imwrite(folder + f'{args.save_name}/k1.rend_side.jpg', (rend * 255).astype(np.uint8)[:, :, ::-1])

    print('all done')



if __name__ == '__main__':
    from argparse import ArgumentParser
    from config.config_loader import load_configs
    parser = ArgumentParser()
    parser.add_argument('exp_name')
    parser.add_argument('-s', '--seq_folder', help="path to one test sequence with the BEHAVE structure")
    parser.add_argument('-on', '--obj_name', default=None,
                        help='object category name, if not provided, will load from sequence information file')

    args = parser.parse_args()

    configs = load_configs(args.exp_name)
    configs.seq_folder = args.seq_folder
    configs.obj_name = args.obj_name
    configs.save_name = "demo"
    configs.redo = False

    # default settings
    configs.batch_size = 1.0
    configs.filter_val = 0.004
    configs.sparse_thres = 0.03
    configs.batch_size = 1
    configs.start = 0
    configs.end = None

    main(configs)