"""
generate dense point cloud from test image

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import torch
from glob import glob
import numpy as np
from torch.nn import functional as F
import sys, os
from os.path import isfile, join
sys.path.append(os.getcwd())
from model import CHORE


class Generator:
    def __init__(self, model: CHORE,
                 exp_name,
                 threshold=1.0,
                 checkpoint=None,
                 device=torch.device("cuda"),
                 multi_gpus=True,
                 sparse_thres=0.05,
                 filter_val=0.03
                 ):
        self.sparse_thres = sparse_thres
        self.filter_val = filter_val
        self.sample_num = 100000
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold

        self.multi_gpus = multi_gpus  # model trained with multi-gpus or not, for loading checkpoint
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format(
            exp_name)  # use new path
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        assert os.path.exists(self.checkpoint_path), f'{self.checkpoint_path} does not exist!'
        self.load_checkpoint(checkpoint)
        for param in self.model.parameters():
            param.requires_grad = False

        # for visualization
        bmax = np.array([3.0, 1.80, 4.0])
        bmin = np.array([-3.0, -0.9, 0.2])
        self.pmin = bmin
        self.pmax = bmax

    def approx_surface(self, model, samples, num_steps, query_input, df_type):
        """
        iteratively project query point to the surface, represented by a neural UDF
        :param model:
        :param samples:
        :param num_steps:
        :param query_input:
        :param df_type:
        :return:
        """
        df_idx = 0 if df_type == 'human' else 1
        preds = None
        for j in range(num_steps):
            model.query(samples, **query_input)
            preds = model.get_preds()
            df_pred, pca_pred, parts_pred = preds[:3]  # first three is always the same data
            df_target = torch.clamp(df_pred[:, df_idx, :], max=self.threshold)

            df_target.sum().backward()

            gradient = samples.grad.detach()  # compute gradients for the sample points
            samples = samples.detach()

            # update sample points to approximate surface point,    alg 1 in the paper
            samples = samples - F.normalize(gradient, dim=2) * df_target.unsqueeze(-1)

            samples = samples.detach()
            samples.requires_grad = True

        return samples, preds

    def get_grid_samples(self, sample_num, batch_size=1):
        "samples in a 3d grid"
        return self.init_samples(sample_num, batch_size)

    def filter(self, data):
        "encode image features"
        images = data['images'].to(self.device)
        self.model.filter(images)

    def parse_preds(self, batch_size, counts, mask, out_dict, out_names, preds, samples_surface):
        """
        save network predicts to out_dict
        """
        for i in range(batch_size):
            # add points
            out_dict['points'][i].append(samples_surface[i, mask[i]].detach().cpu())
            # handle each example separately
            for name, pred in zip(out_names[1:], preds[1:]):
                out_dict[name][i].append(pred[i, ..., mask[i]].detach().cpu())
            counts.append(torch.sum(mask[i]).item())  # count how many new points are added

    def generate_pclouds_batch(self, data, num_steps=10,
                             num_points=50000, mute=False):
        """
        generate point clouds for a batch if input
        :param data: a batch if input data from dataloader
        :param num_steps:
        :param num_points: minimum number of surface points generated for each example
        :param mute: if True, do not print generation process
        :return:
        """
        self.filter(data)
        batch_size = data.get('images').shape[0]
        samples = self.get_grid_samples(30000, batch_size=batch_size)
        targets = ['human', 'object']
        data_dict = {}
        for t in targets:
            data_dict[t] = self.gen_pc_batch(self.model, t,
                                       samples, num_points, data, num_steps, mute=mute)

        return data_dict

    def gen_pc_batch(self, model, df_type, samples_init, num_points, batch, num_steps, max_iter=100, mute=False):
        """
        core function to iteratively project points to the surface.
        :param model:
        :param df_type: UDF of human or object
        :param samples_init: init query points
        :param num_points: target number of point, stop if enough surface points found
        :param batch: data batch from data loader
        :param num_steps: projection steps from query point to surface
        :param max_iter: allowable maximum number of iterations, raise error if generation did not finish within max_iter
        :param mute: do not predict intermediate point generation information
        :return:
        """
        query_input = self.prep_query_input(batch)
        df_idx = 0 if df_type == 'human' else 1
        batch_size = samples_init.shape[0]

        out_names = ['points', 'pca_axis', 'parts', 'centers']
        out_dict = {}
        for n in out_names:
            out_dict[n] = [[] for x in range(batch_size)]
        sample_num = 20000

        iter, samples_count = 0, 0
        samples = samples_init.clone().to(self.device)
        samples.requires_grad = True
        while samples_count < num_points:
            samples_surface, preds = self.approx_surface(model, samples, num_steps,
                                                         query_input, df_type=df_type)
            df_pred = preds[0]
            df_target = torch.clamp(df_pred[:, df_idx, :], max=self.threshold).detach()
            mask = df_target < self.filter_val
            # collect predictions for points close to surface
            if iter > 0:
                counts = []
                self.parse_preds(batch_size, counts, mask, out_dict, out_names, preds, samples_surface)
                samples_count += np.min(counts)
                if not mute:
                    print("{} points".format(samples_count))

            # get samples for next iteration
            samples_new = []
            for i in range(batch_size):
                samples_i = samples[i, mask[i], :].unsqueeze(0)
                if samples_i.shape[1] > 1:
                    indices = torch.randint(samples_i.shape[1], (1, sample_num))
                    samples_i = samples_i[[[0, ] * sample_num], indices]

                    # perturb the samples
                    samples_i += (self.threshold / 3) * torch.randn(samples_i.shape).to(self.device)
                else:
                    indices = torch.randint(samples_init.shape[1], (1, sample_num))
                    samples_i = torch.tensor(samples_init[[[i, ] * sample_num], indices]).to(self.device)
                    samples_i += 0.5* torch.randn(1, sample_num, 3).to(self.device)

                samples_new.append(samples_i)
            samples_new = torch.cat(samples_new, 0)
            samples = samples_new.detach()
            samples.requires_grad = True

            iter += 1
            if iter == max_iter:
                raise RuntimeError('point generation failed after 100 iterations')

        self.compose_outdict(batch_size, out_dict, out_names, samples_count)
        return out_dict

    def compose_outdict(self, batch_size, out_dict, out_names, samples_count):
        """
        compose a batch of all neural predictions for points around the surface
        :param batch_size:
        :param out_dict:
        :param out_names:
        :param samples_count: min number of points in this batch
        :return: a dict of points: (B, N, 3), predicted part labels: (B, N), object rotation (pca_axis): (B, 3, 3),
        smpl and object center (B, 6)
        """
        for name in out_names:
            out_batch = out_dict[name]
            batch_comb = []
            for i in range(batch_size):
                if name == 'points':
                    out_i = torch.cat(out_batch[i], 0)  # points: (N, 3)
                    batch_comb.append(out_i[:samples_count, :])
                    continue
                out_i = torch.cat(out_batch[i], -1)
                out_i = out_i[..., :samples_count]
                if name == 'parts':
                    out_i = torch.argmax(out_i, 0)
                elif name == 'pca_axis':
                    out_i = torch.mean(out_i, -1)
                elif name == 'centers':
                    out_i = torch.mean(out_i, -1)  # compute the average centers
                batch_comb.append(out_i)
            out_dict[name] = torch.stack(batch_comb, 0)

    def get_val_min_ck(self):
        file = glob(self.exp_path + 'val_min=*')
        if len(file) == 0:
            return None
        log = np.load(file[0])
        path = self.checkpoint_path + str(log[2])
        if not isfile(path):
            return None
        # print('Found best checkpoint:', path)
        return log[2]

    def find_best_checkpoint(self, checkpoints):
        """if val min is presented, use that to find, otherwise use the latest one"""
        val_min_ck = self.get_val_min_ck()
        if val_min_ck is None:
            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
                *[*convertSecs(checkpoints[-1]), checkpoints[-1]])
        else:
            path = self.checkpoint_path + val_min_ck
        return path

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path + '/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0, 0
            path = self.find_best_checkpoint(checkpoints)
        else:
            path = self.checkpoint_path + '{}'.format(checkpoint)
        checkpoint = torch.load(path)
        print('Loaded checkpoint from: {}'.format(path))

        state_dict = {}
        if self.multi_gpus:
            # load checkpoint saved by distributed data parallel
            for k, v in checkpoint['model_state_dict'].items():
                newk = k.replace('module.', '')
                state_dict[newk] = v
        else:
            state_dict = checkpoint['model_state_dict']

        self.model.load_state_dict(state_dict)
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        return epoch, training_time

    def prep_query_input(self, batch):
        crop_center = batch.get('crop_center').to(self.device)  # (B, 3)
        return {
            'crop_center': crop_center
        }

    def init_samples(self, sample_num, batch_size=1):
        "sample around fixed smpl center"
        samples = torch.rand(batch_size, sample_num, 3).float().to(self.device)  # generate random sample points
        samples[0, :, 0] = samples[0, :, 0] * 6 - 3  # x: -3, 3
        samples[0, :, 1] = samples[0, :, 1] * 5 - 2.5  # y: -2.5, 2.5
        samples[0, :, 2] = (samples[0, :, 2] - 0.5 )*0.5  + 2.2 # z: 1.95 - 2.45

        return samples


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds
