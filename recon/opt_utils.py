"""
common util functions for optimization

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""

from psbody.mesh import Mesh
import pickle as pkl
import os.path as osp
import cv2
import numpy as np

# 14 body part colors
mturk_colors = np.array(
    [44, 160, 44,
     31, 119, 180,
     255, 127, 14,
     214, 39, 40,
     148, 103, 189,
     140, 86, 75,
     227, 119, 194,
     127, 127, 127,
     189, 189, 34,
     255, 152, 150,
     23, 190, 207,
     174, 199, 232,
     255, 187, 120,
     152, 223, 138]
).reshape((-1, 3))/255.

# path to the simplified mesh used for registration
_mesh_template = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}


def get_template_path(behave_path, obj_name):
    return osp.join(behave_path, _mesh_template[obj_name])

def load_scan_centered(scan_path, cent=True):
    """load a scan and centered it around origin"""
    scan = Mesh()
    # print(scan_path)
    scan.load_from_file(scan_path)
    if cent:
        center = np.mean(scan.v, axis=0)

        verts_centerd = scan.v - center
        scan.v = verts_centerd

    return scan


def save_smplfits(save_paths,
                  scores,
                  smpl,
                  save_mesh=True,
                  ext='.ply'):
    verts, _, _, _ = smpl()
    verts_np = verts.cpu().detach().numpy()
    B = verts.shape[0]
    faces_np = smpl.faces.cpu().detach().numpy()
    for i in range(B):
        v = verts_np[i, :, :]
        f = faces_np
        if save_mesh:
            mesh = Mesh(v=v, f=f)
            if save_paths[i].endswith('.ply'):
                mesh.write_ply(save_paths[i])
            else:
                mesh.write_obj(save_paths[i])
    save_smpl_params(smpl, scores, save_paths, ext=ext)


def save_smpl_params(smpl, scores, mesh_paths, ext='.ply'):
    poses = smpl.pose.cpu().detach().numpy()
    betas = smpl.betas.cpu().detach().numpy()
    trans = smpl.trans.cpu().detach().numpy()
    for p, b, t, s, n in zip(poses, betas, trans, scores, mesh_paths):
        smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'score': s}
        pkl.dump(smpl_dict, open(n.replace(ext, '.pkl'), 'wb'))
    return poses, betas, trans, scores


def mask2bbox(mask):
    "convert mask to bbox in xyxy format"
    ret, threshed_img = cv2.threshold(mask,
                                      127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bmin, bmax = np.array([50000, 50000]), np.array([-100, -100])
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        bmin = np.minimum(bmin, np.array([x, y]))
        bmax = np.maximum(bmax, np.array([x+w, y+h]))
    return np.concatenate([bmin, bmax], 0) # xyxy format