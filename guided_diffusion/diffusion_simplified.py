import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download

import torchvision.utils as tvu
from guided_diffusion.diffusion import *

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.celeba import CelebA
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np
import torchvision
from PIL import Image
from functools import partial


def simplified_ddnm_plus(self, model, cls_fn):
    args, config = self.args, self.config

    #dataset, test_dataset = get_dataset(args, config)
    test_dataset = CombinedDataset(args, config)

    device_count = torch.cuda.device_count()

    if args.subset_start >= 0 and args.subset_end > 0:
        assert args.subset_end > args.subset_start
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
    else:
        args.subset_start = 0
        args.subset_end = len(test_dataset)

    print(f'Dataset has size {len(test_dataset)}')

    def seed_worker(worker_id):
        worker_seed = args.seed % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    val_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # get degradation operator
    print("args.deg:",args.deg)
    if args.deg =='colorization':
        A = lambda z: color2gray(z)
        Ap = lambda z: gray2color(z)
    elif args.deg =='denoising':
        A = lambda z: z
        Ap = A
    elif args.deg =='sr_averagepooling':
        scale=round(args.deg_scale)
        A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
        Ap = lambda z: MeanUpsample(z,scale)
    elif args.deg =='inpainting':
        loaded = np.load("exp/inp_masks/mask.npy")
        mask = torch.from_numpy(loaded).to(self.device)
        A = lambda z: z*mask
        Ap = A
    elif args.deg =='mask_color_sr':
        loaded = np.load("exp/inp_masks/mask.npy")
        mask = torch.from_numpy(loaded).to(self.device)
        A1 = lambda z: z*mask
        A1p = A1
        
        A2 = lambda z: color2gray(z)
        A2p = lambda z: gray2color(z)
        
        scale=round(args.deg_scale)
        A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
        A3p = lambda z: MeanUpsample(z,scale)
        
        A = lambda z: A3(A2(A1(z)))
        Ap = lambda z: A1p(A2p(A3p(z)))
    elif args.deg =='diy':
        # design your own degradation
        loaded = np.load("exp/inp_masks/mask.npy")
        mask = torch.from_numpy(loaded).to(self.device)
        A1 = lambda z: z*mask
        A1p = A1
        
        A2 = lambda z: color2gray(z)
        A2p = lambda z: gray2color(z)
        
        scale=args.deg_scale
        A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
        A3p = lambda z: MeanUpsample(z,scale)
        
        A = lambda z: A3(A2(A1(z)))
        Ap = lambda z: A1p(A2p(A3p(z)))
    else:
        raise NotImplementedError("degradation type not supported")

    args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
    sigma_y = args.sigma_y
    
    print(f'Start from {args.subset_start}')
    idx_init = args.subset_start
    idx_so_far = args.subset_start
    avg_psnr = 0.0
    pbar = tqdm.tqdm(val_loader)
    for idx, (x_orig, classes), (x_deg,_) in pbar:
        x_orig = x_orig.to(self.device)
        x_deg= x_deg.to(self.device)
        x_orig = data_transform(self.config, x_orig)
        x_deg = data_transform(self.config, x_deg)

        y = x_deg #A_funcs.A(x_orig)
        
        h = x_deg.size(2)
        w = x_deg.size(3)
        c = x_deg.size(1)
        b = x_deg.size(0)
        hw = h*w
        hwc = h*w*c
        
        y = y.reshape((b, hwc))

        if config.sampling.batch_size!=1:
            raise ValueError("please change the config file to set batch size as 1")

        #Apy = Ap(y)

        os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        for i in range(x_orig.shape[0]):
            """
            tvu.save_image(
                inverse_data_transform(config, Apy[i]),
                os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
            )
            """
            tvu.save_image(
                inverse_data_transform(config, x_orig[i]),
                os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
            )
            
        # init x_T
        x = torch.randn(
            y.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        with torch.no_grad():
            skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
            n = x.size(0)
            x0_preds = []
            xs = [x]
            
            times = get_schedule_jump(config.time_travel.T_sampling, 
                                            config.time_travel.travel_length, 
                                            config.time_travel.travel_repeat,
                                            )
            time_pairs = list(zip(times[:-1], times[1:]))
            
            
            # reverse diffusion sampling
            for i, j in tqdm.tqdm(time_pairs):
                i, j = i*skip, j*skip
                if j<0: j=-1 

                if j < i: # normal sampling 
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = (1 - at_next**2).sqrt()
                    xt = xs[-1].to('cuda')

                    et = model(xt, t)

                    if et.size(1) == 6:
                        et = et[:, :3]

                    # Eq. 12
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # Eq. 19
                    if sigma_t >= at_next*sigma_y:
                        lambda_t = 1.
                        gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                    else:
                        lambda_t = (sigma_t)/(at_next*sigma_y)
                        gamma_t = 0.

                    # Eq. 17
                    x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                    eta = self.args.eta

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    # different from the paper, we use DDIM here instead of DDPM
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))    
                else: # time-travel back
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t = x0_preds[-1].to('cuda')

                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                    xs.append(xt_next.to('cpu'))

            x = xs[-1]
            
        x = [inverse_data_transform(config, xi) for xi in x]

        tvu.save_image(
            x[0], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
        )
        orig = inverse_data_transform(config, x_orig[0])
        mse = torch.mean((x[0].to(self.device) - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        avg_psnr += psnr

        idx_so_far += y.shape[0]

        pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

    avg_psnr = avg_psnr / (idx_so_far - idx_init)
    print("Total Average PSNR: %.2f" % avg_psnr)
    print("Number of samples: %d" % (idx_so_far - idx_init))
