"""
Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from os.path import isfile
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, device,
                 train_dataset,
                 val_dataset,
                 exp_name,
                 optimizer='Adam',
                 lr=1e-3,
                 threshold=0.1,
                 multi_gpus=False,
                 rank=-1,
                 world_size=-1,
                 input_type='RGB',
                 **kwargs):
        self.model = model
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)
        self.start_lr = lr # starting learning rate
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, kwargs.get('milestones'), 0.3)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        print(f'{len(train_dataset)} training and {len(val_dataset)} validation examples.')
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.val_min = self.load_val_min()
        self.max_dist = threshold
        self.input_type = input_type

        self.multi_gpus = multi_gpus
        if multi_gpus:
            assert rank >=0, "if use multi gpu, make sure rank and world size is passed in"
            self.rank = rank
            self.world_size=world_size
        else:
            self.rank=0
            self.world_size=1 # for single GPU training
        if self.rank==0:
            # only allow master process to write log
            self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))

            # keep track of loss, manually
            self.train_losses = [0.0]*4 # binary, parts, clamped df, surface df
            self.train_counts = [0]*4 # one for binary, parts, clamped df, and the other for surface df
            self.val_losses = [0.0] * 4  # binary, parts, clamped df, surface df
            self.val_counts = [0] * 4

        self.ck_period = kwargs.get('ck_period')

    def train_step(self,batch):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss, sep_error = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        # return loss.item()
        return loss.item(), sep_error

    def compute_loss(self, batch):
        if self.multi_gpus:
            # multiple gpu: use nn.module
            points = batch.get('points').cuda(self.rank,
                                              non_blocking=True)  # for multiple GPUs, must set non_blocking and explicitly tell the device
            df_h = batch.get('df_h').cuda(self.rank, non_blocking=True)  # (Batch,num_points)
            df_o = batch.get('df_o').cuda(self.rank, non_blocking=True)  # (Batch,num_points)
            parts_gt = batch.get('labels').cuda(self.rank, non_blocking=True).long()  # (B, N)
            pca_gt = batch.get('pca_axis').cuda(self.rank, non_blocking=True)  # (Batch,num_points)
            input_images = batch.get('images').cuda(self.rank, non_blocking=True)  # (B, F, H, W)

            body_center = batch.get('body_center').cuda(self.rank, non_blocking=True)  # (B, 2)
            crop_center = batch.get('crop_center').cuda(self.rank, non_blocking=True)  # (B, 2)
            obj_center = batch.get('obj_center').cuda(self.rank, non_blocking=True)  # (B, 3)
            error = self.model(input_images, points, df_h,
                               df_o,
                               parts_gt,
                               pca_gt,
                               max_dist=self.max_dist,
                               body_center=body_center,
                               # offsets=offsets,
                               obj_center=obj_center,
                               crop_center=crop_center,
                               )
            return error

        else:
            device = self.device
            points = batch.get('points').to(device)
            df_gt = batch.get('df').to(device)  # (Batch,num_points)
            labels_gt = batch.get('labels').to(device)  # (B, N)
            images = batch.get('image').to(device)

            df_pred, parts_pred = self.model(images, points)

            loss_df = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=self.max_dist),
                                                        torch.clamp(df_gt,
                                                                    max=self.max_dist))  # out = (B,num_points) by componentwise comparing vecots of size num_samples:
            loss_df = loss_df.sum(
                -1).mean()  # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
            loss_label = F.cross_entropy(parts_pred, labels_gt.long(),
                                         reduction='none') * 0.02  # TODO: tune this parameter
            loss_label = loss_label.sum(-1).mean()  # first batch: label loss=269, df loss=175

            return loss_label + loss_df

    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader(rank=self.rank,
                                                          world_size=self.world_size)
        print("start training model with total {} training AND {} testing examples...".format(len(self.train_dataset),
                                                                                              len(self.val_dataset)))
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()
        print("starting from epoch:", start)
        for epoch in tqdm(range(start, epochs)):
            sum_loss = 0
            # print('Node {} Start epoch {}'.format(self.rank, epoch))
            loop = tqdm(train_data_loader)
            loop.set_description("rank {}: ".format(self.rank))
            separate_losses = 0.
            batch_count = 0
            for batch in loop:
                iteration_duration = time.time() - iteration_start_time
                if iteration_duration > self.ck_period * 60:  # eve model every X min and at start
                    training_time += iteration_duration
                    iteration_start_time = time.time()  # update for next saving period
                    self.eval_model(training_time, epoch)

                    # also log losses
                    if self.rank == 0 and batch_count > 0:
                        # N = len(train_data_loader)
                        self.writer.add_scalar('training loss steps', sum_loss / batch_count,
                                               batch_count + epoch * len(train_data_loader))
                        self.writer.add_scalars('train losses steps', self.format_separate_loss(separate_losses,
                                                                                                batch_count),
                                                batch_count + epoch * len(train_data_loader))

                # optimize model
                loss = self.train_step(batch)
                err, sep_err = loss
                sum_loss += err
                separate_losses += sep_err
                batch_count += 1
            # print("Last batch loss: {}".format(err / self.train_dataset.total_sample_num))
            if self.rank == 0:
                # N = len(train_data_loader)
                self.writer.add_scalar('training loss last batch', err, epoch)
                self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
                self.writer.add_scalars('train losses', self.format_separate_loss(separate_losses,
                                                                                  len(train_data_loader)), epoch)
                self.writer.add_scalars('lr', {f'param{i}': v for i, v in enumerate(self.lr_scheduler.get_lr())}, epoch)
            # synchronize multi-gpus
            torch.cuda.synchronize(self.rank)
            self.eval_model(training_time, epoch)  # evaluate model in each step
            self.lr_scheduler.step()

        self.eval_model(training_time, epochs)
        self.save_checkpoint(epoch, training_time)  # save model at the end of training

    def save_checkpoint(self, epoch, training_time):
        filename = 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(training_time), training_time])
        path = self.checkpoint_path + filename
        if not os.path.exists(path):
            if not self.multi_gpus:
                torch.save({  # 'state': torch.cuda.get_rng_state_all(),
                    'training_time': training_time, 'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)
            else:
                # multiple gpus, only device 0 save the model
                if self.rank==0:
                    torch.save({  # 'state': torch.cuda.get_rng_state_all(),
                        'training_time': training_time, 'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)
                    print("checkpoint {} saved".format(path))

        else:
            print("{} already exist, not saving, time{}".format(path, datetime.now()))
        return filename

    def get_val_min_ck(self):
        file = glob(self.exp_path + 'val_min=*')
        if len(file) == 0:
            return None
        log = np.load(file[0])
        path = self.checkpoint_path + str(log[2])
        if not isfile(path):
            return None
        print('find best checkpoint:', path)
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

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0.

        path = self.find_best_checkpoint(checkpoints)

        print('Loaded checkpoint from: {}'.format(path))

        if self.multi_gpus:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        else:
            map_location = None

        checkpoint = torch.load(path, map_location=map_location)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        L1 = len(checkpoint['optimizer_state_dict']['param_groups'][0]['params'])
        L2 = len(self.optimizer.param_groups[0]['params'])
        print(L1, L2)
        if L1 == L2:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: inconsistent param groups, not loading optimizer state dict. ck params: {}, current optimizer params: {}".format(
                L1, L2
            ))
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        for g in self.optimizer.param_groups:
            g['lr'] = self.start_lr
            print("setting lr to {}".format(self.start_lr))

        return epoch, training_time

    def compute_val_loss(self):
        self.model.eval()
        val_loader = self.val_dataset.get_loader(shuffle=True)  # evaluation not using  multi-gpu
        sum_val_loss = 0.0
        num_batches = min(64, len(val_loader))

        count = 0
        val_loop = tqdm(val_loader)
        val_loop.set_description('validation loop:')
        separate_losses = 0.
        for val_batch in val_loop:
            with torch.no_grad():  # speed up computation and save memory
                error, sep_errors = self.compute_loss(val_batch)
                sum_val_loss += error.item()
                count = count + 1
                separate_losses += sep_errors
            if count >= num_batches:
                break

        print("Finished evaluation")
        if count != 0:
            err = sum_val_loss / count
            print("error: {}".format(err))
            return err, separate_losses / count
        else:
            return np.inf, np.inf

    def eval_model(self, training_time, epoch):
        "evaluate model"
        # self.save_checkpoint(epoch, training_time)
        if self.rank == 0:
            # only master process do the evaluation
            val_loss, separate_losses = self.compute_val_loss()
            if self.val_min is None:
                self.val_min = val_loss

            ck_file = self.save_checkpoint(epoch, training_time)  # always save the best model
            if val_loss <= self.val_min + 1.0:  # the longer, the better
                self.val_min = val_loss  # update the best model
                self.update_vmin_file(ck_file, epoch, val_loss)

            self.writer.add_scalar('val loss batch avg', val_loss, epoch)
            self.writer.add_scalars('val losses', self.format_separate_loss(separate_losses, 1), epoch)

    def update_vmin_file(self, ck_file, epoch, val_loss):
        for path in glob(self.exp_path + 'val_min=*'):
            os.remove(path)
        # later use this info to load the best checkpoint
        np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss, ck_file])

    def load_val_min(self):
        file = glob(self.exp_path + 'val_min=*')
        if len(file) == 0:
            return None
        log = np.load(file[0])
        return float(log[1])

    def format_separate_loss(self, separate_losses, length):
        "format so that it can be sent to tensorboard"
        loss_dict = {}
        if len(separate_losses) == 2:
            for name, sp_loss in zip(['smpl', 'obj'], separate_losses):
                loss_dict[name] = sp_loss / length
        elif len(separate_losses) == 3:
            for name, sp_loss in zip(['smpl', 'obj', 'pca'], separate_losses):
                loss_dict[name] = sp_loss / length
        else:
            for name, sp_loss in zip(['df_h', 'df_o', 'parts', 'pca', 'smpl', 'obj'], separate_losses):
                loss_dict[name] = sp_loss / length
        return loss_dict


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
