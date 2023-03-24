import time
import argparse
import glob
import os
import random
import sys
from glaze.utils import get_parameters, load_img
import torch
from glaze.glazeopt import GlazeOptimizer
from pathlib import Path
PROD = True
ENABLE_GPU = True
home_path = Path.home()
PROJECT_ROOT_PATH = os.path.join(home_path, '.glaze')
if not os.path.exists(PROJECT_ROOT_PATH):
    os.makedirs(PROJECT_ROOT_PATH)
    TARGET_LIST = [
        'impressionism painting by van gogh;0.55']

class Glaze(object):
    
    def __init__(self, mode, opt_mode, gpu, strength, exp_name, jpg, output_dir):
        self.params = get_parameters(mode, opt_mode)
        self.gpu = gpu
        self.strength = strength
        self.exp_name = exp_name
        self.project_root_path = PROJECT_ROOT_PATH
        self.mode = mode
        self.opt_mode = opt_mode
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, True, **('exist_ok',))
            self.device = self.detect_device()
        self.target_params = self.load_target_info()
        self.optimizer = GlazeOptimizer(self.params, self.device, self.target_params, self.project_root_path, jpg, **('project_root_path', 'jpg'))
        self.optimizer.output_dir = self.output_dir
        print('device: ', self.device)

    
    def update_params(self, output_dir, p_level, compute_level, signal):
        print(p_level, compute_level)
        self.signal = signal
        p_level = p_level
        if compute_level == -1:
            compute_level = '0'
        elif compute_level == 0:
            compute_level = '1'
        elif compute_level == 1:
            compute_level = '2'
        elif compute_level == 2:
            compute_level = '3'
        else:
            raise Exception('Unknown render level of {:.2f}'.format(compute_level))
        self.params = None(p_level, compute_level)
        self.target_params = self.load_target_info()
        self.output_dir = output_dir
        self.optimizer.params = self.params
        self.optimizer.output_dir = output_dir
        self.optimizer.target_params = self.target_params
        self.optimizer.signal = self.signal
        if compute_level != '0':
            if output_dir == 'not selected':
                raise Exception('Please select output folder before proceeding. ')
            if not None.path.exists(output_dir):
                raise Exception('Output folder {} does not exist. '.format(output_dir))
            return None

    def load_target_info(self):
        if not PROD:
            target_params = {
                'style': 'impressionism painting by van gogh',
                'strength': self.strength,
                'seed': 3242}
            return target_params

        target_file = os.path.join(self.project_root_path, 'target.txt')

        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                data = f.read().split('\n')
                idx = int(data[0])
                seed = int(data[1])
        else:
            idx = random.choice(range(len(TARGET_LIST)))
            seed = random.randrange(1, 1000)
            data = "{}\n{}".format(idx, seed)
            with open(target_file, 'w+') as f:
                f.write(data)

        cur_target = TARGET_LIST[idx]
        cur_style, cur_strength = cur_target.split(';')
        cur_strength = float(cur_strength)

        if self.params['opt_setting'] == '0':
            cur_actual_strength = 0.6
        else:
            cur_actual_strength = self.cal_strength(cur_strength)

        target_params = {
            'style': cur_style,
            'strength': cur_actual_strength,
            'seed': seed}

        return target_params

    
    def cal_strength(self, cur_strength):
        if self.params['setting'] < 20:
            actual_strength = cur_strength - 0.05
        elif self.params['setting'] < 40:
            actual_strength = cur_strength - 0.01
        elif self.params['setting'] < 60:
            actual_strength = cur_strength + 0.03
        else:
            actual_strength = cur_strength + 0.08
        actual_strength = min(actual_strength, 0.6)
        return actual_strength

    
    def detect_device(self):
        if torch.cuda.is_available():
            t = torch.cuda.get_device_properties(0).total_memory / 1048576
            if t > 5000 and t < 30000:
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = 'cpu'
        print(f'''Run on {device}''')
        return device

    
    def run_protection_prod(self, image_paths):
        s = time.time()
        (out_file_ls, is_error) = self.optimizer.generate(image_paths)
        print('Total time {:.2f}'.format(time.time() - s))
        if is_error:
            raise Exception('Glaze finished. However, Glaze encountered errors for at least one image you input. Please open error.txt in {} for details. Please try increasing the Intensity level and/or render quality. '.format(self.output_dir))

    
    def run_tests(self, image_paths):
        self.optimizer.output_dir = self.output_dir
        out_file_ls = self.optimizer.generate(image_paths)
        return out_file_ls



def main(*argv):
    if not argv:
        argv = list(sys.argv)
        parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str, default='imgs/')
    parser.add_argument('--out-dir', '-od', type=str, default=None)
    parser.add_argument('--mode', '-m', type=str, default='2')
    parser.add_argument('--strength', '-s', type=float, default=0.55)
    parser.add_argument('--opt-length', '-o', type=str, default='2')
    parser.add_argument('--gpu', '-g', type=str, default=None)
    parser.add_argument('--exp', '-e', type=str, default=None)
    parser.add_argument('--jpg', '-j', type=int, default=0)
    args = parser.parse_args(argv[1:])
    image_paths = glob.glob(os.path.join(args.directory, '*'))
    image_paths = [path for path in image_paths if '_cloaked' not in path.split('/')[-1]]
    protector = Glaze(args.gpu, args.mode, args.opt_length, args.strength, args.exp, args.jpg, args.out_dir)
    protector.run_tests(image_paths)
