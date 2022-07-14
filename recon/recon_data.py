"""
Simple reader to read reconstruction results

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
sys.path.append(os.getcwd())
from os.path import join, isfile
from behave.frame_data import FrameDataReader
import pickle as pkl
import cv2


class ReconDataReader(FrameDataReader):
    def __init__(self, recon_path, seq_folder, check_image=False, ext='jpg'):
        """

        :param recon_path: root path to reconstruction results
        :param seq_folder: path to one BEHAVE sequence
        :param check_image: whether to check if one frame contains complete color and depth images, False for non-behave data
        :param ext: rgb image file extension
        """
        super(ReconDataReader, self).__init__(seq_folder, check_image=check_image, ext=ext)
        self.recon_path = recon_path
        self.recon_seq_path = join(recon_path, self.seq_name)

    def get_mask_full(self, idx, kid):
        "load full object mask, without occlusion"
        file = join(self.get_frame_folder(idx), f'k{kid}.obj_rend_full.jpg')
        if not isfile(file):
            return None
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE) > 127
        return mask

    def get_recon_frame_folder(self, idx):
        if isinstance(idx, int):
            assert idx < len(self)
            return join(self.recon_seq_path, self.frames[idx])
        elif isinstance(idx, str):
            return join(self.recon_seq_path, idx)
        else:
            raise NotImplemented

    def get_recon(self, frame, save_name, tid):
        """
        load CHORE reconstruction result of a specific frame
        :param frame:
        :param save_name:
        :param tid:
        :return: reconstructed smpl and object mesh, None if no mesh found
        """
        save_folder = join(self.get_recon_frame_folder(frame), save_name)
        smpl_file = join(save_folder, f'k{tid}.smpl.ply')
        obj_file = join(save_folder, f'k{tid}.object.ply')
        if (not isfile(smpl_file)) or (not isfile(obj_file)):
            return None, None
        smpl_recon = self.load_mesh(smpl_file)
        obj_recon = self.load_mesh(obj_file)
        return smpl_recon, obj_recon

    def load_crop_info(self, idx, tid=1):
        """
        all necessary cropping information for one test image, saved by testloader
        """
        file = join(self.get_frame_folder(idx), f'k{tid}.crop_info.pkl')
        if not isfile(file):
            return None
        data = pkl.load(open(file, 'rb'))
        return data
