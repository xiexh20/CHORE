"""
sample boundary points and compute ground truth labels

if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import numpy as np
import trimesh
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from psbody.mesh import Mesh
import igl


class BoundarySampler:
    def __init__(self):
        pass

    def boundary_sampling(self, smpl, obj, sigma=0.05,
                          sample_num=100000, grid_ratio=0.01):
        """
        sample boundary points on a pair of interacting SMPL and object mesh
        :param smpl: SMPL mesh
        :param obj: object mesh
        :param sigma: Gaussian standard deviation to perturb surface samples
        :param sample_num: total number of samples
        :param grid_ratio: ratio to generate grid point samples
        :return: samples and GT labels
        """
        comb = trimesh.util.concatenate([smpl, obj])

        # get sample points
        boundary_points = comb.sample(sample_num) + sigma * np.random.randn(sample_num, 3)

        # get grid points
        pmin, pmax = BoundarySampler.get_bounds()
        grid_samples = self.get_grid_samples(pmin, pmax, int(grid_ratio*sample_num))
        samples_all = np.concatenate([boundary_points, grid_samples], 0)

        # Get distance from smpl
        temp_h = trimesh.proximity.ProximityQuery(smpl)
        ret = igl.signed_distance(samples_all, smpl.vertices, smpl.faces.astype(int), return_normals=False)
        d_h, _, neighbours_h = ret
        d_h = np.abs(d_h).astype(np.float32)
        neighbours_h = neighbours_h.astype(np.float32)

        # Get distance from object
        ret = igl.signed_distance(samples_all, obj.vertices, obj.faces.astype(int), return_normals=False)
        d_o, _, neighbours_o = ret
        d_o = np.abs(d_o).astype(np.float32)
        neighbours_o = neighbours_o.astype(np.float32)

        # Parts
        part_labels = pkl.load(open('assets/smpl_parts_dense.pkl', 'rb'))
        labels = np.zeros((6890,), dtype='int32')
        for n, k in enumerate(part_labels):
            labels[part_labels[k]] = n

        _, vert_ids = temp_h.vertex(samples_all)
        parts = labels[vert_ids]

        return (
            samples_all,
            d_h,
            d_o,
            parts,
            neighbours_h,
            neighbours_o
        )

    def flip_part_labels(self, parts):
        "flip part labels"
        flip_parts_map = {
            # left to right
            1:6,
            2:7,
            3:8,
            4:9,
            5:10,
            12:13,
            # right to left
            6:1,
            7:2,
            8:3,
            9:4,
            10:5,
            13:12
        }
        new_labels = parts.copy()
        for part in flip_parts_map.keys():
            mask = parts == part
            new_labels[mask] = flip_parts_map[part]
        return new_labels

    def get_sample_num(self, ratio, total_sample, thres=10000):
        sample_num_s = int(ratio*total_sample)
        if sample_num_s < thres:
            return thres
        return sample_num_s

    def boundary_sample_all(self, landmark, smpl_mesh:Mesh, obj_mesh:Mesh,
                            sigmas, ratios, sample_num,
                            grid_ratio=1/16.,
                            flip=False):
        """
        do boundary sampling for a set of different sigmas
        :param landmark: to compute body center
        :param smpl_mesh:
        :param obj_mesh:
        :param sigmas:
        :param ratios: ratios for different sigmas
        :param flip: flipped part label or not
        :return:
        """
        smpl = trimesh.Trimesh(vertices=smpl_mesh.v, faces=smpl_mesh.f, process=False)
        obj = trimesh.Trimesh(vertices=obj_mesh.v, faces=obj_mesh.f, process=False)

        points_all, dh_all, do_all, parts_all, grad_h, grad_o = {}, {}, {}, {}, {}, {}
        neighbours_h_all, neighbours_o_all = {}, {}
        for s, r in zip(sigmas, ratios):
            sample_num_s = self.get_sample_num(r, sample_num)
            points, d_h, d_o, parts, n_h, n_o = self.boundary_sampling(smpl, obj, s, sample_num_s, grid_ratio=grid_ratio)
            points_all['sigma{}'.format(s)] = points.astype(np.float32)
            dh_all['sigma{}'.format(s)] = d_h.astype(np.float32)
            do_all['sigma{}'.format(s)] = d_o.astype(np.float32)
            if flip:
                parts = self.flip_part_labels(parts)
            parts_all['sigma{}'.format(s)] = parts.astype(np.uint8)

            neighbours_h_all['sigma{}'.format(s)] = n_h.astype(np.float32)
            neighbours_o_all['sigma{}'.format(s)] = n_o.astype(np.float32)
        pca_axis = BoundarySampler.compute_pca(obj)

        # also save smpl center
        body_center = landmark.get_smpl_center(smpl_mesh)
        body_kpts = landmark.get_body_kpts(smpl_mesh)

        # save object center
        obj_center = np.mean(obj_mesh.v, 0)

        data_dict = {
            'points': points_all,
            'dist_h': dh_all,
            'dist_o': do_all,
            'parts': parts_all,
            'pca_axis': pca_axis.astype(np.float32),
            'smpl_center': body_center,
            'body_kpts': body_kpts.astype(np.float32),
            'obj_center': obj_center.astype(np.float32),
        }

        return data_dict

    @staticmethod
    def compute_pca(obj):
        "compute pca axis for the object mesh, obj is trimesh object"
        pca = PCA(n_components=3)
        pca.fit(obj.vertices)
        pca_axis = pca.components_
        return pca_axis

    def get_grid_samples(self, pmin, pmax, sample_num):
        points = np.random.rand(sample_num, 3)
        prange = pmax - pmin
        for i in range(3):
            points[:, i] = points[:, i] * prange[i] + pmin[i]
        return points

    @staticmethod
    def get_bounds():
        """
        fixed grid bounds
        """
        bmax = np.array([3.0, 1.80, 4.0])
        bmin = np.array([-3.0,  -0.9,  0.2])
        return bmin, bmax


