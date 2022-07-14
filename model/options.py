import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')

        # my input arguments
        g_data.add_argument('--dataset_path', type=str, default='/BS/xxie-3/static00/newdata',
                            help='path to dataset')
        g_data.add_argument('--exp_name', type=str, default="train")
        g_data.add_argument('--test_kid', type=int, default=1)
        g_data.add_argument('--image_size', default=(2048, 1536), help="original image size")
        g_data.add_argument('--net_img_size', nargs='+', type=int, default=[256,192], help="image size sent to network")
        g_data.add_argument("--focal_length", default=(1000, 1000), help="focal length for the perspective camera")
        g_data.add_argument("--subfolder_name", default='recon_data', help='name for the subfolder in a dataset')
        g_data.add_argument("--depth2color", default=True, help='perform depth to color transform or not', action='store_true')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--batch_size', type=int, default=8, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epochs', type=int, default=100, help='num epoch to train')
        g_train.add_argument('--num_samples_train', type=int, default=5000, help='number of training samples used for training')
        g_train.add_argument('--multi_gpus', default=True, action='store_true', help="whether multiple GPUs are used during training")
        g_train.add_argument("--split_file", default="/BS/xxie2020/work/kindata/Apr20/frames/split.json",
                             help="the file specifying how to split train and test sequence")
        g_train.add_argument("--clamp_thres", type=float, default=0.1,
                             help="the threshold to clamp df prediction when computing loss")
        g_train.add_argument('--mix_samp', default=True, action='store_true')
        g_train.add_argument("--sigmas", default=[0.08, 0.02, 0.003], nargs='+', type=float,
                             help="gaussian variance fou boundary sampling, unit meter")
        g_train.add_argument("--person_obj_ratio", default=[0.5, 0.5], nargs='+', type=float,
                             help="ratio for person and object points")
        g_train.add_argument('--ratios', type=float, nargs='+', default=[0.01, 0.49, 0.5],
                             help='ratio between different sigmas')
        g_train.add_argument("--clean_only", default=False, help='use only clean data or not', action='store_true')

        # arguments from PIFu
        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')
        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')

        g_exp.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
        g_exp.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')


        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')

        # for multiple gpu training
        g_train.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        g_train.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')

        # g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
        g_train.add_argument('--num_workers', default=30, type=int, help="number of thres for loading data")
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        

        g_train.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
       
        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        g_train.add_argument('--scan_data', action='store_true', default=False)
        g_train.add_argument('--data_name', type=str)

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')
        g_test.add_argument("--eval_num", type=int, default=10,
                            help='number of examples to evaluate')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=5.0, help='perturbation standard deviation for positions')

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')
        g_sample.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')
        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')
        g_sample.add_argument('--realdepth', default=False, action='store_true', help="input real depth to the network or inverse")

        g_sample.add_argument("--densepc_num", type=int, default=10000, help='number of dense point cloud to generate at evaluation time')

        # Model related
        g_model = parser.add_argument_group('Model')
        g_model.add_argument("--model_type", default='comb', help='which model to use for training')
        g_model.add_argument("--input_type", default="RGBM", help="RGB, RGB+D, RGB+normal")
        g_model.add_argument('--num_parts', default=15, type=int, help='number of output part labels')
        g_model.add_argument('--encode_type', default='normal_hg')
        g_model.add_argument("--surface_classifier", default=False, help='use surface classification or nor', action='store_true')
        g_model.add_argument('--joint_df', default=False, help='joint distance field for human and object', action='store_true')

        # for anchor UDF
        parser.add_argument('--reso_grid', type=int, default=32, help='# resolution of grid')
        parser.add_argument('--pn_hid_dim', type=int, default=32, help='# hidden dim of point net')
        parser.add_argument('--num_anchor_points', type=int, default=600, help='number of anchor points')

        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--norm_color', type=str, default='instance',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--bin_classifier', default=True, action='store_true',
                             help='use binary classifier or not')

        # hg filter specify
        # g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_stack', type=int, default=3, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General==>> not needed anymore
        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp') # 257 = 256 (output from HGFilter) + 1 (depth feature)
        g_model.add_argument('--mlp_dim_color', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')

        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')

        # for train
        parser.add_argument('--random_flip', action='store_true', help='if random flip')
        parser.add_argument('--random_trans', action='store_true', help='if random flip')
        parser.add_argument('--random_scale', action='store_true', help='if random flip')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1')
        # z_normalization
        parser.add_argument('--z_feat', help='which z feature is sent to the network')

        # camera mode
        parser.add_argument('--projection_mode', default='perspective', type=str)
        parser.add_argument('--orth_size', type=int, default=512)
        parser.add_argument('--orth_scale', type=float, default=0.75)

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')
        parser.add_argument('--filter_val', type=float, default=0.004,
                            help='threshold to filter out points not on the surface')
        parser.add_argument('--sparse_thres', type=float, default=0.03,
                            help='threshold to get sparse pc around the surface')
        parser.add_argument('--save_densepc', action='store_true',
                            help="save generated dense pc during evaluation or not, do not save when evaluate over all data")
        parser.add_argument('--save_npz', action='store_true', default=False,
                            help='save npz during dense pc generation or not')
        parser.add_argument('--pcsave_name', default=None, help='name to save for this experiment evaluation')
        parser.add_argument("--seq_folder", type=str, help="which sequence to evaluate")
        parser.add_argument('--checkpoint', type=str, default=None, help='which checkpoint to load for evaluation')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask')
        parser.add_argument('--img_path', type=str, help='path for input image')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')
        group_aug.add_argument('--nocrop', default=False, action='store_true')

        # special tasks
        self.initialized = True

        # to write config file
        parser.add_argument('--overwrite', default=False, action='store_true')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
