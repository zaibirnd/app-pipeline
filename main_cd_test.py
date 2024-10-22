# from argparse import ArgumentParser
# import torch
# from models.trainer import *
# import os
# import shutil
# import time
# import datetime
# import numpy as np
# import glob
# import argparse
# import concurrent.futures
# from Cropping_Thread import CroppingThread


print(torch.cuda.is_available())

import warnings
warnings.filterwarnings("ignore")

"""
the main function for training the CD networks
"""
# root_for_qgis = os.path.expanduser('~') + '/bda/CD_pipeline_multi/'
root_for_qgis = os.getcwd()+'/'

def exists_rm_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
def preparing_crops(args):
    image_paths = glob.glob(args.input + '/*.tif')
    print('*****************',image_paths)
    print(f'[INFO]: Cropping process in progress...')
    tc = time.time()
    cropping_thread = CroppingThread(args)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(cropping_thread.slicethread, image_paths)

def test(args):
    from models.evaluator_test import CDEvaluator
    dataloaders = utils.get_loader(args.data_name, args ,img_size=args.img_size,dataset=args.dataset,
                                  batch_size=1, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloaders)
    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # preprocess
    # parser.add_argument('--root', type=str, default=os.path.expanduser('~') + '/bda/CD_pipeline_multi/')
    parser.add_argument('--root', type=str, default=os.getcwd()+'/')
    # parser.add_argument('--input', type=str, default=root_for_qgis + 'data/images', help='path to the satellite images')
    parser.add_argument('--input', type=str, default='data/images', help='path to the satellite images')
    parser.add_argument('--step_size', type=float, default=0.1, help='stride for image cropping')
    parser.add_argument('--patch_size', type=int, default=1024, help='crop dimension')

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='TestInput', type=str)
    parser.add_argument('--data_name', default='TestInput', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=512, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_05', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8| base_transformer_pos_s4_dd8_t8_e2d4')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')

    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)

    exists_rm_create('vis/')
    args.vis_dir = os.path.join('vis', args.project_name)
    exists_rm_create(args.vis_dir)
    args.geojson = root_for_qgis + 'geojson/'
    exists_rm_create(args.geojson)
    args.test_cases = root_for_qgis + 'data/construction/test_cases/images'
    exists_rm_create(args.test_cases)
    preparing_crops(args)
    test(args)