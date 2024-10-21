
import os
import shutil
import time
import datetime
import numpy as np
import glob
import argparse
import concurrent.futures
from Cropping_Thread import CroppingThread


def preparing_crops(args):
    image_paths = glob.glob(args.input + '/*.tif')
    print(image_paths)
    print(f'[INFO]: Cropping process in progress...')
    tc = time.time()
    cropping_thread = CroppingThread(args)
    print('1')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(cropping_thread.slicethread, image_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/images', help='path to the satellite images')
    parser.add_argument('--step_size', type=float, default=0.1, help='stride for image cropping')
    parser.add_argument('--patch_size', type=int, default=1024, help='crop dimension')
    args = parser.parse_args()
    preparing_crops(args)
    



