import gc
from PIL import Image
import random
import torch
import numpy as np
import time
import glob
from glaze.utils import load_img, img2tensor, tensor2img, CLIP, check_clip_threshold
import torchvision
from torchvision import transforms
from diffusers.models.vae import DiagonalGaussianDistribution
import os
import pickle
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
import shutil
PROD = True
BATCH_SIZE = 1
EVAL = False
LOG = False
CHECK_RUN = False
PERFORMANCE = True
DEBUG = False

class GlazeOptimizer(object):
    
    def __init__(self, params, device, target_params, project_root_path, jpg):
        self.params = params
        self.device = device
        self.jpg = jpg
        self.output_dir = None
        if device in ('cuda', 'mps'):
            self.half = True
        else:
            self.half = False
        self.target_params = target_params
        self.project_root_path = project_root_path
        self.stable_diffusion_model = None
        self.num_segments_went_through = 0

    
    def load_encoder_models(self):
        self.model = torch.load(os.path.join(self.project_root_path, 'glaze.p'), torch.device('cpu'), **('map_location',)).to(self.device).to(torch.float32)
        self.model_qc = torch.load(os.path.join(self.project_root_path, 'glaze-qc.p'), torch.device('cpu'), **('map_location',)).to(self.device).to(torch.float32)
        if self.half:
            self.model = self.model.half()
            self.model_qc = self.model_qc.half()
            return None

    
    def unload_encoder_models(self):
        self.model.to('cpu')
        del self.model
        self.model_qc.to('cpu')
        del self.model_qc
        gc.collect()


    def clean_tmp(self):
        if PERFORMANCE or self.params['opt_setting'] == '0':
            return None
        tmp_files = glob.glob(os.path.join(self.project_root_path, 'tmp/*'))
        for f in tmp_files:
            os.remove(f)
        target_files = glob.glob(os.path.join(self.project_root_path, 'target-*.jpg'))
        for f in target_files:
            os.remove(f)
        return None


    
    def load_eval_models(self):
        import clip
        self.clip_model = CLIP(self.device, self.project_root_path)
        self.full_vae = AutoencoderKL.from_pretrained(os.path.join(self.project_root_path, 'base', 'base'), 'vae', **('subfolder',))
        self.full_vae.to(self.device)
        if self.half:
            self.full_vae = self.full_vae.half()
            return None

    
    def model_encode(self, input_tensor):
        h = self.model(input_tensor)
        moments = self.model_qc(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.mean

    
    def generate(self, image_paths):
        error_logs = [
            'Below are errors Glaze encountered while protecting your art: \n']
        self.num_segments_went_through = 0
        final_path_ls = []
        all_tmp_paths = []
        self.tot_runs = self.params['n_runs']
        image_paths = [f for f in image_paths if '-glazed-intensity' not in f]
        image_paths = sorted(image_paths)
        legit_image_paths = []
        image_data_ls = []
        for f in image_paths:
            cur_img = load_img(f, self.project_root_path)
            if cur_img is None:
                print('ERROR loading {}'.format(f))
                continue
            legit_image_paths.append(f)
            image_data_ls.append(cur_img)
        if len(legit_image_paths) == 0:
            raise Exception('Zero image detected')
        if not None(legit_image_paths) > 20 and PERFORMANCE:
            raise Exception('Glaze can only process at most 20 images at a time. ')
        if None(legit_image_paths) != 1 and self.params['opt_setting'] == '0':
            raise Exception('You can only preview one image at a time. Select only one image for preview. ')
        self.extract_all_targets(image_data_ls)
        if self.params['opt_setting'] != '0':
            self.load_encoder_models()
        for run_num in range(self.params['n_runs']):
            self.cur_run = run_num
            (cur_tmp_paths, legit_image_paths) = self.generate_one_run(image_data_ls, legit_image_paths, run_num)
            all_tmp_paths.append(cur_tmp_paths)
        if self.params['opt_setting'] != '0':
            self.unload_encoder_models()
        if self.params['opt_setting'] != '0':
            self.load_eval_models()
            self.signal.emit('display=Glaze generated, evaluating strength')
            for idx in range(len(all_tmp_paths[0])):
                cur_seed = random.randrange(0, 1000)
                og_file_path = legit_image_paths[idx]
                og_score = self.cal_clip_score(og_file_path, cur_seed)
                cur_best_score = -1
                cur_best_path = None
                print(og_file_path.split('/')[-1])
                for run_num in range(self.params['n_runs']):
                    cur_image_path = all_tmp_paths[run_num][idx]
                    cur_score = self.cal_clip_score(cur_image_path, cur_seed)
                    if cur_score > cur_best_score:
                        cur_best_score = cur_score
                        cur_best_path = cur_image_path
                    if PERFORMANCE:
                        print(run_num, cur_score - og_score)
                        continue
                    og_img = Image.open(og_file_path)
                    cur_meta = og_img.getexif()
                    final_path = self.move_image(cur_best_path, og_file_path, cur_meta)
                    clip_diff = cur_best_score - og_score
                    print('B', clip_diff, cur_best_score)
                    is_success = check_clip_threshold(self.params, clip_diff)
                    if not is_success:
                        error_code = '{:.4f}'.format(clip_diff)
                        error_code = error_code.split('.')[-1]
                        error_logs.append('Warning: we failed to produce strong enough protection for this art ({}). Please try increasing the Intensity level and/or render quality. ERROR CODE: {}'.format(og_file_path.split('/')[-1], error_code))
                        error_path = 'INCOMPLETE-' + os.path.basename(final_path)
                        error_path = os.path.join(self.output_dir, error_path)
                        if os.path.exists(error_path):
                            os.remove(error_path)
                        shutil.move(final_path, error_path)
                        final_path = error_path
                final_path_ls.append(final_path)
        is_error = False

        outf = os.path.join(self.output_dir, 'error.txt')

        if os.path.exists(outf):
            os.remove(outf)

        if len(error_logs) > 1:
            text = '\n'.join(error_logs)
            with open(outf, 'w+') as f:
                f.write(text)
            is_error = True

        self.clean_tmp()

        return final_path_ls, is_error


    
    def cal_clip_score(self, img_path, seed):
        image_transforms = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([
                0.5], [
                0.5])])
        cur_img = load_img(img_path, self.project_root_path)
        random_r = random.randrange(0, 100)
        cur_tmp_path = os.path.join(self.project_root_path, 'foo_{}.jpg'.format(random_r))
        cur_img.save(cur_tmp_path, 93, **('quality',))
        cur_img = Image.open(cur_tmp_path)
        torch.manual_seed(seed)
        with torch.no_grad():
            tensor_img = image_transforms(cur_img).unsqueeze(0)
            tensor_img = tensor_img[:, :, :512, :512]
            tensor_img = tensor_img.to(self.device)
            if self.half:
                tensor_img = tensor_img.half()
                cur_res_img = self.full_vae(tensor_img).sample
            cur_res_img = tensor2img(cur_res_img)
            cur_score = self.clip_model(cur_res_img, self.target_params['style'])

        os.remove(cur_tmp_path)
        return cur_score

    
    def move_image(self, tmp_path, og_file_path, meta):
        target_file_path = self.cal_target_file_name(og_file_path, False, 0)
        if os.path.exists(target_file_path):
            os.remove(target_file_path)
            tmp_img = Image.open(tmp_path)
        tmp_img.save(target_file_path, meta, 100, **('exif', 'quality'))
        return target_file_path

    def generate_one_run(self, image_data_ls, legit_image_paths, run_num):
        final_path_ls = []
        self.tot_num_imgs = len(image_data_ls)

        for idx in range(len(image_data_ls)):
            if self.params['opt_setting'] == '0':
                self.signal.emit('display=generating preview for image {} / {}'.format(idx + 1, len(image_data_ls)))
            else:
                self.signal.emit('display=glazing image {} / {}'.format(idx + 1, len(image_data_ls)))

            cur_img = image_data_ls[idx]
            cur_target = Image.open(os.path.join(self.project_root_path, 'target-{}-{}.jpg'.format(idx, run_num)))

            if cur_target.size != cur_img.size:
                raise Exception("Target and source size mismatch")

            self.cur_img_idx = idx
            res_cloaked_imgs = self.generate_one_image(cur_img, [cur_target])

            if len(res_cloaked_imgs) != 1:
                raise AssertionError

            res_cloaked_img = res_cloaked_imgs[0]
            og_image_path = legit_image_paths[idx]

            tmp_path = self.save_image(res_cloaked_img, og_image_path, True, run_num, tmp='run_num')
            final_path_ls.append(tmp_path)

        return final_path_ls, legit_image_paths

    
    def save_image(self, cur_cloak_img, og_image_path, tmp, run_num = (False, 0)):
        glaze_fpath = self.cal_target_file_name(og_image_path, tmp, run_num)
        if cur_cloak_img.size[0] == 512 and cur_cloak_img.size[1] == 512:
            glaze_fpath += '.png'
            if glaze_fpath.endswith('.png'):
                cur_cloak_img.save(glaze_fpath, 'PNG', **('format',))
            else:
                cur_cloak_img.save(glaze_fpath, 'JPEG', 100, **('format', 'quality'))
        return glaze_fpath

    
    def cal_target_file_name(self, og_image_path, tmp, run_num):
        if tmp:
            cur_dir = os.path.join(self.project_root_path, 'tmp')
            os.makedirs(cur_dir, True, **('exist_ok',))
        elif self.output_dir is not None and os.path.exists(self.output_dir):
            cur_dir = self.output_dir
        else:
            raise Exception('Cannot locate output folder')
        og_file_name = None.path.basename(og_image_path)
        if '.' in og_file_name:
            og_file_name_first = '.'.join(og_file_name.split('.')[:-1])
            og_file_name_last = og_file_name.split('.')[-1]
        else:
            og_file_name_first = og_file_name
            og_file_name_last = None
        glazed_file_name_first = og_file_name_first + '-glazed-intensity{}-render{}'.format(self.params['setting'], self.params['opt_setting'] if self.params['opt_setting'] != '0' else '-preview')
        if tmp:
            glazed_file_name_first = glazed_file_name_first + '-run{}'.format(run_num)
            if og_file_name_last is None:
                glazed_file_name = glazed_file_name_first
            else:
                glazed_file_name = glazed_file_name_first + '.' + og_file_name_last
        glaze_fpath = os.path.join(cur_dir, glazed_file_name)
        return glaze_fpath

    def generate_one_image(self, img_ls, target_ls):
        assert len(img_ls) == 1
        assert len(target_ls) == 1

        all_segments = []
        all_target_segments = []
        og_image_arrays = []
        square_size_ls = []
        square_size_dup_ls = []
        imgidx2segment = {}
        imgidx2lastidx = {}

        for idx, cur_img in enumerate(img_ls):
            cur_target_img = target_ls[idx]
            cur_img_array = np.array(cur_img).astype(np.float32)

            og_image_arrays.append(cur_img_array)

            segments, cur_last_idx, cur_square_size = self.segement_img(cur_img)
            cur_target_segment, _, _ = self.segement_img(cur_target_img)

            square_size_ls.append(cur_square_size)
            square_size_dup_ls += [cur_square_size] * len(segments)

            imgidx2lastidx[idx] = cur_last_idx
            imgidx2segment[idx] = len(all_segments)

            all_segments += segments
            all_target_segments += cur_target_segment

        res_adv_list = self.compute_512_adv_tensor(all_segments, all_target_segments, square_size_dup_ls)

        res_cloaked_imgs = []

        for idx in range(len(img_ls)):
            og_img_array = og_image_arrays[idx]
            segment_idx = imgidx2segment[idx]
            square_size = square_size_ls[idx]

            cur_cloak_list = [res_adv_list[i] for i in range(segment_idx, segment_idx + square_size)]
            last_index = imgidx2lastidx[idx]

            cloaked_array = self.put_back_cloak(og_img_array, cur_cloak_list, last_index)
            cloaked_img = Image.fromarray(cloaked_array.astype(np.uint8))

            res_cloaked_imgs.append(cloaked_img)

        return res_cloaked_imgs

    
    def extract_all_targets(self, image_data_ls):
        for idx, cur_img in enumerate(image_data_ls):
            self.signal.emit('display=analyzing images (~{} min per image)...'.format(3 * self.params['n_runs']))
            for r in range(self.params['n_runs']):
                cur_target_img = self.style_transfer(cur_img)
                cur_target_img.save(os.path.join(self.project_root_path, 'target-{}-{}.jpg'.format(idx, r)))
            if self.stable_diffusion_model is not None:
                self.stable_diffusion_model.to('cpu')
                del self.stable_diffusion_model
                self.stable_diffusion_model = None
                gc.collect()
                return None

    
    def compute_512_adv_tensor(self, all_segments, all_target_segments, square_size_dup_ls):
        res_adv_list = []
        for seg_idx in range(len(all_segments)):
            if len(all_segments) == 1:
                raise AssertionError()
            cur_square_size = square_size_dup_ls[seg_idx]
            seg_start_time = time.time()

            res_adv_tensors = self.compute_batch(
                all_segments[seg_idx : seg_idx + len(all_segments)],
                all_target_segments[seg_idx : seg_idx + len(all_segments)],
                cur_square_size,
                seg_idx,
                len(all_segments),
            )

            seg_processing_time = time.time() - seg_start_time
            print("process time:", seg_processing_time)

            if seg_processing_time < 40:
                if not PERFORMANCE and self.params["opt_setting"] != "0":
                    time.sleep(50 - seg_processing_time)

                for cur_adv_tensor in res_adv_tensors:
                    res_adv_list.append(cur_adv_tensor)

        return res_adv_list

    
    def compute_batch(self, cur_batch, target_batch, cur_square_size, seg_idx, tot_seg_length):
        cur_batch = [img2tensor(img, self.device) for img in cur_batch]
        cur_targets = [img2tensor(img, self.device) for img in target_batch]
        max_change = self.params['max_change']
        tot_steps = self.params['tot_steps']

        if self.half:
            source_image_tensors = [x.half() for x in cur_batch]
            target_image_tensors = [x.half() for x in cur_targets]
        else:
            source_image_tensors = cur_batch
            target_image_tensors = cur_targets

        source_batch = torch.cat(source_image_tensors, axis=0)
        target_batch = torch.cat(target_image_tensors, axis=0)

        if CHECK_RUN or self.params['opt_setting'] == '0':
            preview_mask = pickle.load(open(os.path.join(self.project_root_path, 'preview_mask.p'), 'rb'))
            preview_mask_tensor = torch.tensor(preview_mask, dtype=source_batch.dtype).to(self.device)
            preview_mask_tensor = (preview_mask_tensor / 0.05) * max_change * 1.2
            cloaked_batch = source_batch + preview_mask_tensor
            cloaked_batch = torch.clamp(cloaked_batch, -1, 1)
            return cloaked_batch

        rand_amount = None
        X_batch = source_batch.clone().detach().to(self.device)

        with torch.no_grad():
            target_emb = self.model_encode(target_batch).detach()

        resizerlarge = torchvision.transforms.Resize(cur_square_size)
        resizer512 = torchvision.transforms.Resize((512, 512))
        pbar = range(tot_steps)

        if tot_steps > 10:
            step_size = max_change * 0.5
        else:
            step_size = max_change * 0.75

        best_modifier = None
        modifiers = torch.zeros_like(X_batch)

        for i in pbar:
            actual_step_size = step_size * (1 - (100 * (i / tot_steps)))
            tot_seg_length2 = 2 * self.tot_runs * self.tot_num_imgs
            cur_p = ((i / tot_steps) * 100 * (tot_seg_length2 / self.num_segments_went_through)) + (100 * (tot_seg_length2 / self.tot_num_imgs))
            self.signal.emit(f'glazetp={cur_p:.2f}')

            modifiers.requires_grad_(True)
            X_adv = torch.clamp(modifiers + X_batch, -1, 1)
            X_adv = resizerlarge(X_adv)
            X_adv = resizer512(X_adv)

            loss_normal = (self.model_encode(X_adv) - target_emb).norm()
            tot_loss = loss_normal
            grad = torch.autograd.grad(tot_loss, modifiers)[0]

            grad = grad.detach()
            modifiers = modifiers.detach()

            final_cur_update = grad.sign() * actual_step_size
            modifiers = modifiers - final_cur_update

            modifiers = torch.clamp(modifiers, -max_change, max_change)
            grad = None
            best_modifier = modifiers.detach().cpu().numpy()

        best_modifier_t = torch.tensor(best_modifier).to(self.device).half()
        best_adv_tensors = torch.clamp(best_modifier_t + X_batch, -1, 1)

        self.num_segments_went_through += 1

        return best_adv_tensors

    
    def load_model(self):
        model_path = os.path.join(self.project_root_path, 'base', 'base')
        m = StableDiffusionImg2ImgPipeline.from_pretrained(model_path)
        m.to(self.device)
        m.enable_attention_slicing()
        return m

    
    def get_cloak(self, og_segment_img, res_adv_tensor, square_size):
        resize_back_og_img = og_segment_img.resize((square_size, square_size))
        res_adv_img = tensor2img(res_adv_tensor).resize((square_size, square_size))
        cur_cloak = np.array(res_adv_img).astype(np.float32) - np.array(resize_back_og_img).astype(np.float32)
        return cur_cloak

    
    def segement_img(self, cur_img):
        cur_img_array = np.array(cur_img).astype(np.float32)
        (og_width, og_height) = cur_img.size
        short_height = og_height <= og_width
        if short_height:
            squares_ls = []
            last_index = 0
            cur_idx = 0
            square_size = og_height
            if cur_idx + og_height < og_width:
                cropped_square = cur_img_array[0:og_height, cur_idx:cur_idx + og_height, :]
            else:
                cropped_square = cur_img_array[0:og_height, -og_height:, :]
                last_index = og_height - og_width - cur_idx
            cropped_square = Image.fromarray(cropped_square.astype(np.uint8))
            cropped_square = cropped_square.resize((512, 512))
            squares_ls.append(cropped_square)
            cur_idx = cur_idx + og_height
            if cur_idx >= og_width:
                pass
            
        else:
            squares_ls = []
            last_index = 0
            cur_idx = 0
            square_size = og_width
            if cur_idx + og_width < og_height:
                cropped_square = cur_img_array[cur_idx:cur_idx + og_width, 0:og_width, :]
            else:
                cropped_square = cur_img_array[-og_width:, 0:og_width, :]
                last_index = og_width - og_height - cur_idx
            cropped_square = Image.fromarray(cropped_square.astype(np.uint8))
            cropped_square = cropped_square.resize((512, 512))
            squares_ls.append(cropped_square)
            cur_idx = cur_idx + og_width
            if cur_idx >= og_height:
                pass
            
        return (squares_ls, last_index, square_size)

    
    def put_back_cloak(self, og_img_array, cloak_list, last_index):
        (og_height, og_width, _) = og_img_array.shape
        short_height = og_height <= og_width
        if short_height:
            for idx, cur_cloak in enumerate(cloak_list):
                if idx < len(cloak_list) - 1:
                    og_img_array[0:og_height, idx * og_height:(idx + 1) * og_height, :] += cur_cloak
                    continue
                og_img_array[0:og_height, idx * og_height:(idx + 1) * og_height, :] += cur_cloak[0:og_height, last_index:]
        else:
            for idx, cur_cloak in enumerate(cloak_list):
                if idx < len(cloak_list) - 1:
                    og_img_array[idx * og_width:(idx + 1) * og_width, 0:og_width, :] += cur_cloak
                    continue
                og_img_array[idx * og_width:(idx + 1) * og_width, 0:og_width, :] += cur_cloak[last_index:, 0:og_width]
        og_img_array = np.clip(og_img_array, 0, 255)
        return og_img_array

    def style_transfer(self, cur_img):
        if self.params['opt_setting'] == '0' or CHECK_RUN:
            return cur_img
        if self.stable_diffusion_model is None:
            self.stable_diffusion_model = self.load_model()
            if self.params['opt_setting'] == '0':
                n_run = 10
            else:
                n_run = self.params['style_transfer_iter']
        prompts = [self.target_params['style']]
        strength = self.target_params['strength']
        img_copy = cur_img.copy()
        img_copy.thumbnail((512, 512), Image.ANTIALIAS)
        canvas = np.zeros((512, 512, 3)).astype(np.uint8)
        canvas[:img_copy.size[1], :img_copy.size[0], :] += np.array(img_copy)
        cropped_target_img = None

        if cropped_target_img is None:
            padded_img = Image.fromarray(canvas)
            img_tensor = img2tensor(padded_img, device=self.device)
            with torch.no_grad():
                target_img = self.stable_diffusion_model(prompts, img_tensor, strength, 7.5, n_run).images
            target_img = target_img[0]
            cropped_target_img = np.array(target_img)[:img_copy.size[1], :img_copy.size[0], :]
            cropped_target_img = Image.fromarray(cropped_target_img)
            full_target_img = cropped_target_img.resize(cur_img.size)
            return full_target_img


