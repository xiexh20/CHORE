"""
if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import pickle as pkl
from os.path import join

from psbody.mesh import Mesh
from scipy import sparse
import torch


def load_regressors(assets_root, batch_size=None):
    body25_reg = pkl.load(open(join(assets_root, 'body25_regressor.pkl'), 'rb'), encoding="latin1").T
    face_reg = pkl.load(open(join(assets_root, 'face_regressor.pkl'), 'rb'), encoding="latin1").T
    hand_reg = pkl.load(open(join(assets_root, 'hand_regressor.pkl'), 'rb'), encoding="latin1").T
    if batch_size is not None:
        # convert to torch tensor
        body25_reg_torch = torch.sparse_coo_tensor(body25_reg.nonzero(), body25_reg.data, body25_reg.shape)
        face_reg_torch = torch.sparse_coo_tensor(face_reg.nonzero(), face_reg.data, face_reg.shape)
        hand_reg_torch = torch.sparse_coo_tensor(hand_reg.nonzero(), hand_reg.data, hand_reg.shape)

        return torch.stack([body25_reg_torch] * batch_size), torch.stack([face_reg_torch] * batch_size), \
               torch.stack([hand_reg_torch] * batch_size)
    return body25_reg, face_reg, hand_reg


class BodyLandmarks:
    "SMPL wrapper to compute body landmarks with SMPL meshes"
    def __init__(self, assets_root):
        br, fr, hr = load_regressors(assets_root)
        self.body25_reg = br
        self.face_reg = fr
        self.hand_reg = hr
        self.parts_inds = self.load_parts_ind(p=join(assets_root, 'smpl_parts_dense.pkl'))

    def get_landmarks(self, smpl_mesh:Mesh):
        """
        return keyjoints of body, face and hand
        """
        verts = smpl_mesh.v

        body = sparse.csr_matrix.dot(self.body25_reg, verts)
        face = sparse.csr_matrix.dot(self.face_reg, verts)
        hand = sparse.csr_matrix.dot(self.hand_reg, verts)

        return body, face, hand

    def get_smpl_center(self, smpl_mesh:Mesh):
        verts = smpl_mesh.v
        body = sparse.csr_matrix.dot(self.body25_reg, verts)

        return body[8]

    def load_parts_ind(self, p='assets/smpl_parts_dense.pkl'):
        d = pkl.load(open(p, 'rb'))
        return d

    def get_body_kpts(self, smpl_mesh:Mesh):
        "return all 25 body keypoints"
        verts = smpl_mesh.v
        body = sparse.csr_matrix.dot(self.body25_reg, verts)
        return body

    def get_part_verts(self, smpl, part_name):
        """
        smpl: vertices
        return the vertices for the given part (IP-Net part convension).
        """
        ind = self.parts_inds[part_name]
        return smpl[ind].copy()


def test_parts():
    landmark = BodyLandmarks()
    file = "/BS/xxie2020/work/hoi3d/debug/smpl/restpose_x0_y0_z0.ply"
    mesh = Mesh()
    mesh.load_from_file(file)

    names = ['right_forearm', 'left_forearm']
    colors = ['red', 'green']

    parts_meshes = []
    for n, c in zip(names, colors):
        v = landmark.get_part_verts(mesh.v, n)
        m = Mesh(v, [], vc=c)
        parts_meshes.append(m)

    from psbody.mesh import MeshViewer
    mv = MeshViewer()
    parts_meshes.append(mesh)
    mv.set_static_meshes(parts_meshes)


if __name__ == '__main__':
    test_parts()

