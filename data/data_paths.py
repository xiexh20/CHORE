import glob
import os, re
import pickle as pkl
from os.path import join, basename, dirname, isfile

import cv2, json
import numpy as np

import yaml, sys
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
PROCESSED_PATH = paths['PROCESSED_PATH']


class DataPaths:
    """
    class to handle path operations based on BEHAVE dataset structure
    """
    def __init__(self):
        pass

    @staticmethod
    def load_splits(split_file):
        dataset_path = PROCESSED_PATH
        assert os.path.exists(dataset_path), f'the given dataset path {dataset_path} does not exist, please check if your training data are placed over there!'
        train, val = DataPaths.get_train_test_from_pkl(split_file)
        train_full = [join(dataset_path, x) for x in train] # full path to the training data
        val_full = [join(dataset_path, x) for x in val] # full path to the validation data files
        return train_full, val_full

    @staticmethod
    def get_train_test_from_pkl(pkl_file):
        data = pkl.load(open(pkl_file, 'rb'))
        return data['train'], data['test']

    @staticmethod
    def get_image_paths_seq(seq, tid=1, check_occlusion=False):
        """
        find all image paths in one sequence
        :param seq: path to one behave sequence
        :param tid: test on images from which camera
        :param check_occlusion: whether to load full object mask and check occlusion ratio
        :return: a list of paths to test image files
        """
        image_files = sorted(glob.glob(seq + f"/*/k{tid}.color.jpg"))
        # print(image_files)
        if not check_occlusion:
            return image_files
        # check object occlusion ratio
        valid_files = []
        count = 0
        for img_file in image_files:
            mask_file = img_file.replace('.color.jpg', '.obj_rend_mask.jpg')
            full_mask_file = img_file.replace('.color.jpg', '.obj_rend_full.jpg')
            if not isfile(mask_file) or not isfile(full_mask_file):
                continue

            mask = np.sum(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) > 127)
            mask_full = np.sum(cv2.imread(full_mask_file, cv2.IMREAD_GRAYSCALE) > 127)
            if mask_full == 0:
                count += 1
                continue

            ratio = mask / mask_full
            if ratio > 0.3:
                valid_files.append(img_file)
            else:
                count += 1
                print(f'{mask_file} occluded by {1 - ratio}!')
        return valid_files






