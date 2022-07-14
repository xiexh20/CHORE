"""
save and load configuration

"""
import os, sys
sys.path.append(os.getcwd())
import json
from os.path import join, exists

configs_dir = "config"
def save_configs(args, overwrite=False):
    exp_name = args.exp_name
    filename = join(configs_dir, exp_name+'.json')

    if exists(filename) and not overwrite:
        print('{} already exists'.format(filename))
        raise ValueError

    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("configs saved to {}".format(filename))


def load_configs(exp_name):
    # parser = ArgumentParser()
    from argparse import Namespace
    args = Namespace()
    filename = join(configs_dir, exp_name + '.json')
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    print("configs loaded from {}".format(filename))
    return args

"""
first save the config then train network 
"""
if __name__ == '__main__':
    from model.options import BaseOptions
    args = BaseOptions().parse()
    save_configs(args, overwrite=args.overwrite)
