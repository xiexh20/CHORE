"""
evaluate reconstruction results
Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
sys.path.append(os.getcwd())
from psbody.mesh import Mesh
import trimesh
import numpy as np
from os.path import join, basename
from datetime import datetime
import json
import multiprocessing as mp

from recon.recon_data import ReconDataReader
from recon.eval.pose_utils import ProcrusteAlign
from recon.eval.chamfer_distance import chamfer_distance


class ReconEvaluator:
    def __init__(self, recon_path,
                 dataset_path,
                 smpl_name='fit02',
                 obj_name='fit01',
                 smpl_only=False):
        self.align = ProcrusteAlign(smpl_only=smpl_only)
        self.recon_path = recon_path
        self.dataset_path = dataset_path

        # GT data config: names of the GT data
        self.smpl_name = smpl_name
        self.obj_name = obj_name

        self.outdir = 'results/'
        os.makedirs(self.outdir, exist_ok=True)

        self.manager = mp.Manager()
        self.errors_dict = self.manager.dict()
        self.sample_num = 10000  # number of samples used in computing chamfer distance

    def extract_objname(self, seq_name):
        name = seq_name.split("_")[2]
        return name

    def eva_seq(self, seq, save_name, tid, etype='ours', smpl_only=False):
        reader = ReconDataReader(self.recon_path, seq)
        seq_end = reader.cvt_end(None)
        check_occ = True
        errors_all = []
        for i in range(0, seq_end, 1):
            if check_occ:
                obj_mask = reader.get_mask(i, tid, 'obj')
                mask_full = reader.get_mask_full(i, tid)
                if obj_mask is None or mask_full is None or np.sum(mask_full) == 0:
                    continue
                if np.sum(obj_mask) / np.sum(mask_full) < 0.30:
                    continue

            smpl_fit = reader.get_smplfit(i, self.smpl_name)
            obj_fit = reader.get_objfit(i, self.obj_name)
            if None in [smpl_fit, obj_fit]:
                continue
            gt_meshes = [smpl_fit, obj_fit]
            if etype=='ours':
                smpl_recon, obj_recon = reader.get_recon(i, save_name, tid)
                recon_meshes = [smpl_recon, obj_recon]
            elif etype == 'mocap':
                smpl_recon = reader.get_mocap_mesh(i, tid)
                recon_meshes = [smpl_recon]
                gt_meshes = [smpl_fit]
            else:
                raise NotImplemented
            if smpl_recon is None:
                continue
            # if smpl_only: # only evaluate SMPL mesh
            #     recon_meshes = recon_meshes[:1]
            #     gt_meshes = gt_meshes[:1]
            try:
                recon_aligned = self.align.align_meshes(gt_meshes, recon_meshes)
            except Exception as e:
                print('failed on {}, error: {}'.format(reader.get_frame_folder(i), e))
                continue
            errors = self.compute_errors(gt_meshes, recon_aligned)
            errors_all.append(errors)
        errors_all = np.array(errors_all)
        if len(errors_all) > 0:
            self.errors_dict[basename(seq)] = errors_all
        print(f'{seq} done')

    def compute_errors(self, gt_meshes, aligned):
        """
        compute chamfer distance between surface point samples
        :param gt_meshes: ground truth SMPL and object mesh
        :param aligned: reconstructed SMPL and object mesh, after procrustes alignment
        :return: chamfer distance of human and object
        """
        gt_points = [self.surface_sampling(x) for x in gt_meshes]
        aligned_points = [self.surface_sampling(x) for x in aligned]
        errors = []
        for gt, recon in zip(gt_points, aligned_points):
            err = self.chamfer_dist(gt, recon)
            # err = self.v2v_err(gt, recon)
            errors.append(err)
        errors.append(0.)
        return errors

    def surface_sampling(self, m:Mesh):
        "sample points on the surface"
        m = self.to_trimesh(m)
        points = m.sample(self.sample_num)
        return points

    def to_trimesh(self, m:Mesh):
        "psbody mesh to trimesh"
        trim = trimesh.Trimesh(m.v, m.f, process=False)
        return trim

    def chamfer_dist(self, p1, p2, metric='l2', direction='bi'):
        "chamfer distance error"
        return chamfer_distance(p1, p2, metric, direction)

    def comb_verts(self, meshes):
        comb = []
        for m in meshes:
            comb.append(m.v)
        return np.concatenate(comb, 0)

    def v2v_err(self, p1, p2):
        "vertex to vertex error, p1, p2: (N, 3)"
        return np.sqrt(((p1 - p2) ** 2).sum(axis=-1)).mean(axis=-1)

    def eval_seqs(self, pat, save_name, tid, etype='ours', identifier=''):
        seqs = json.load(open(pat))['seqs']
        smpl_only = identifier == 'smpl'

        jobs = []
        print("evaluating on {} seqs".format(len(seqs)))
        for seq in seqs:
            p = mp.Process(target=self.eva_seq, args=(join(self.dataset_path, seq),
                                                      save_name, tid, etype, smpl_only))
            p.start()
            jobs.append(p)
        for job in jobs:
            job.join()

        self.collect_results(etype, identifier, pat, save_name, tid)

    def collect_results(self, etype, identifier, pat, save_name, tid):
        errors_all = []
        res_all = {}
        res_obj_specific = {}
        # now accumulate errors
        for seq, errors in self.errors_dict.items():
            errors_all.append(errors)
            res_all[seq] = self.format_errors(errors)
            name = self.extract_objname(seq)
            if not name in res_obj_specific:
                res_obj_specific[name] = errors
            else:
                res_obj_specific[name] = np.concatenate([res_obj_specific[name], errors], 0)
        errors_all = np.concatenate(errors_all, 0)
        app = '' if identifier is None else identifier
        self.save_results(errors_all, res_all, pat, save_name, f"{save_name}_{app}_k{tid}", etype,
                          res_obj_specific)

    def format_errors(self, errors, print_summary=False):
        "errors: (N, 3)"
        results = {}
        L = errors.shape[1]
        # names = ['smpl', 'obj', 'comb']
        names = ['smpl', 'obj']
        for i in range(len(names)):
            name = names[i]
            avg = np.mean(errors[:, i]).item()
            std = np.std(errors[:, i]).item()
            results[name] = {'mean':avg, 'std':std}
        results['total'] = len(errors)
        if print_summary:
            print(results)
        return results

    def save_results(self, errors, sep_errors, seqs_pattern,
                     save_name, id, etype, obj_specific_err=None):
        prefix = basename(seqs_pattern).replace('*', '').replace('.json', '_')
        outfile = join(self.outdir, f'{prefix}_{id}_{etype}_{self.get_timestamp()}.json')
        result = self.format_errors(errors, print_summary=True)
        result['separate'] = sep_errors
        result['seqs'] = seqs_pattern
        result['save_name'] = save_name
        result['time'] = self.get_timestamp()
        if obj_specific_err is not None:
            for name, errs in sorted(obj_specific_err.items()):
                result[name] = self.format_errors(errs, print_summary=True)
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("evaluation done, results saved to {}.".format(outfile))

    def get_timestamp(self):
        now = datetime.now()
        time_str = now.isoformat()
        time_str = time_str.replace(':', '-')
        return time_str


def main(args):
    import yaml
    with open("PATHS.yml", 'r') as stream:
        paths = yaml.safe_load(stream)
    BEHAVE_PATH = paths['BEHAVE_PATH']
    RECON_PATH = paths['RECON_PATH']

    evaluator = ReconEvaluator(RECON_PATH, BEHAVE_PATH, smpl_only = args.id == 'smpl')
    evaluator.eval_seqs(args.split, args.save_name, args.tid, args.method, args.id)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-split', default='splits/behave-test.json', help='split file, json file contains all sequence names to compute erros')
    parser.add_argument('-sn', '--save_name', default='chore-release')
    parser.add_argument('-t', '--tid', type=int, default=1)
    parser.add_argument('-m', '--method', default='ours')
    parser.add_argument('-i', '--id', help='if set to smpl, evaluate smpl mesh only')

    args = parser.parse_args()

    main(args)

