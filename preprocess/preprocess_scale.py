"""
move smpl model to a fixed distance, scale it such that the projection still align

if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
import numpy as np
sys.path.append(os.getcwd())
from os.path import join, isfile
from tqdm import tqdm
from glob import glob
from behave.frame_data import FrameDataReader
from behave.kinect_transform import KinectTransform
from preprocess.boundary_sampler import BoundarySampler
from lib_smpl.body_landmark import BodyLandmarks

# load outpaths
import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
BEHAVE_PATH = paths['BEHAVE_PATH']


def process_scale(args):
    sampler = BoundarySampler()
    reader = FrameDataReader(args.seq_folder, check_image=True)
    batch_end = reader.cvt_end(args.end)
    outdir = paths['PROCESSED_PATH']
    smpl_name = args.smpl_name
    obj_name = args.obj_name
    landmark = BodyLandmarks(assets_root=paths['SMPL_ASSETS_ROOT'])
    smpl_depth = args.smpl_depth

    kin_transform = KinectTransform(args.seq_folder, kinect_count=reader.kinect_count)

    scale_skipped = 0
    loop = tqdm(range(args.start, batch_end, args.interval))
    loop.set_description(f'{reader.seq_name} {args.start}-{batch_end}')
    for idx in loop:
        smpl_fit = reader.get_smplfit(idx, smpl_name)
        obj_fit = reader.get_objfit(idx, obj_name)

        if smpl_fit is None or obj_fit is None:
            continue
        for kid in args.kids:
            outfolder = join(outdir, reader.seq_name, reader.frame_time(idx))
            os.makedirs(outfolder, exist_ok=True)
            outfile = join(outfolder, f'{reader.frame_time(idx)}_k{kid}_{args.data_name}.npz')
            if isfile(outfile) and not args.redo:
                continue
            elif isfile(outfile):
                os.system(f'rm {outfile}')
            smpl_local = kin_transform.world2color_mesh(smpl_fit, kid)
            obj_local = kin_transform.world2color_mesh(obj_fit, kid)

            if args.flip:
                outfile = outfile.replace('.npz', '_flip.npz')
                smpl_local = kin_transform.flip_mesh(smpl_local)
                obj_local = kin_transform.flip_mesh(obj_local)

            # Depth-aware scaling
            smpl_center = landmark.get_smpl_center(smpl_local)
            scale = smpl_depth / smpl_center[2]
            smpl_local.v = smpl_local.v * scale # smpl always has a fixed distance to the camera
            obj_local.v = obj_local.v * scale # object is always related to smpl

            if scale < 0.6 or scale > 1.5:
                print('Warnning the scale {} maybe invalid! on file {}, skipped'.format(scale, outfile))
                # assert ,
                scale_skipped += 1
                continue

            # for debug: save scaled mesh
            # smpl_local.write_ply(reader.smplfit_meshfile(idx, smpl_name).replace('.ply', '_scaled.ply'))
            # obj_local.write_ply(reader.objfit_meshfile(idx, obj_name).replace('.ply', '_scaled.ply'))

            new_center = landmark.get_smpl_center(smpl_local)
            assert np.abs(new_center[2] - smpl_depth) <= 1e-6, 'found new depth: {}, target depth: {}'.format(new_center, smpl_depth)

            os.makedirs(outfolder, exist_ok=True)
            data_dict = sampler.boundary_sample_all(landmark, smpl_local, obj_local,
                                                    args.sigmas, args.ratios, args.sample_num,
                                                    grid_ratio=args.grid_ratio, flip=args.flip)
            image_file = reader.get_color_files(idx, [kid])[0]
            assert np.abs(data_dict['smpl_center'][2] - smpl_depth) <= 1e-7, 'found new depth: {}, target depth: {}'.format(data_dict['smpl_center'], smpl_depth)
            data_dict['image_file'] = image_file
            data_dict['sigmas'] = np.array(args.sigmas)
            np.savez(outfile, **data_dict)
    print("skipped {} files, all done".format(scale_skipped))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', "--seq_folder")
    parser.add_argument('-dn', "--data_name",
                        help="the name for this data generation, i.e. subfolder name",
                        default='scale')
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument("--sigmas", default=[0.08, 0.02, 0.003], nargs='+', type=float,
                        help="gaussian variance for boundary sampling, unit meter")
    parser.add_argument('--ratios', type=float, default=[0.01, 0.49, 0.5], help='ratio between different sigmas')
    parser.add_argument('-gr', '--grid_ratio', type=float, default=0.01, help='ratio between different sigmas')
    parser.add_argument("--sample_num", default=100000, type=int, help='total number of samples for each sigma')
    parser.add_argument('-sn', '--smpl_name', help='smpl fitting save name', default='fit02')
    parser.add_argument('-on', '--obj_name', help='object fitting save name', default='fit01')
    parser.add_argument('-k', '--kids', default=[0, 1, 2, 3], nargs='+', type=int)
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-i', '--interval', default=1, type=int)
    parser.add_argument('-flip', default=False, action='store_true')
    parser.add_argument('-sd', '--smpl_depth', default=2.20, type=float, help='the fixed smpl center depth')
    parser.add_argument('-a', '--all', default=False, action='store_true', help='process all sequences sequentially')

    args = parser.parse_args()

    if args.all:
        seqs = sorted(glob(BEHAVE_PATH+"/*"))
        print(f'{len(seqs)} sequences found, starting from {seqs[0]}')
        for seq in seqs:
            args.seq_folder = seq
            process_scale(args)
    else:
        process_scale(args)