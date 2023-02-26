#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:30:04 2022

@author: guo.1648
"""

# Referenced from generate_1run_chenqi.py and projector.py

# generate images for all needed classes in one run!
# But here instead of randomly initialize the latent vector z,
# we use the GAN-inversion latent vector projected_w for initialization!!!

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy

import pickle

import copy
from time import perf_counter

import imageio


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
#@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default='results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl', required=True)
#@click.option('--seeds', type=num_range, help='List of random seeds for generation', default='0-5')
#@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--classes_orig', 'class_idxs', type=num_range, help='List of original Class labels', default='267,327')
#@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
#@click.option('--outdir_root', help='Where to save the output images (root dir for generation!)', default='results/generate_inverInit/Birds_128-011200/val/', type=str, required=True, metavar='DIR')
@click.option('--target_root', help='Root dir of Target image files to project to', default='datasets/Birds/', type=str, required=True, metavar='DIR')
@click.option('--map_pkl_dir', help='The location (full name) of corresponding cGAN_to_orig_cls_map_final.pkl', default='datasets/Birds_128_map/cGAN_to_orig_cls_map_final.pkl', type=str, required=True, metavar='DIR')
@click.option('--num-steps',       help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',           help='Random seed for projection', type=int, default=303, show_default=True)
@click.option('--run_projection',  help='Whether run the projection to get w files', type=bool, default=True, show_default=True)
@click.option('--outdir',          help='Where to save the output projected_w (Note: only do the projection&save once!)', default='results/project/Birds_128-011200/', type=str, required=True, metavar='DIR')

def run_projection_chenqi(
    network_pkl: str,
    target_root: str,
    outdir: str,
    run_projection: bool,
    seed: int,
    num_steps: int,
    class_idxs: Optional[List[int]], # List of original Class labels!
    map_pkl_dir: str
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    
    if not run_projection:
        print('**************** NOT running projection!!!')
        return
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    # load the cGAN_cls -> orig_clas map dict (both keys and items are int):
    f_pkl = open(map_pkl_dir,'rb')
    cGAN_to_orig_cls_map_final = pickle.load(f_pkl)
    f_pkl.close()
    
    # Load target images: newly modified by Chenqi:
    #print(class_idxs)
    for orig_cls in class_idxs:
        cGAN_cls = get_key_from_item(orig_cls, cGAN_to_orig_cls_map_final)
        class_idx = cGAN_cls[0]
        
        # get the output dir correspond to this single class! (using orig class label)
        this_outdir = outdir + str(orig_cls)
        os.makedirs(this_outdir, exist_ok=True)
        
        # get the target img names correspond to this single class! (using orig class label)
        this_target_dir = target_root + str(orig_cls) + '/'
        target_fname_list = os.listdir(this_target_dir)
        
        for target_fname in target_fname_list:
            # Load target image.
            target_pil = PIL.Image.open(this_target_dir+target_fname).convert('RGB')
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
            target_uint8 = np.array(target_pil, dtype=np.uint8)
            
            # Optimize projection.
            start_time = perf_counter()
            projected_w_steps = project(
                G,
                target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                num_steps=num_steps,
                device=device,
                verbose=True,
                c=class_idx # newly added by Chenqi
            )
            print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
            
            # Save final projected frame and W vector.
            target_pil.save(f'{this_outdir}/target_'+target_fname.split('.')[0]+'.png')
            projected_w = projected_w_steps[-1] # shape: torch.Size([12, 512]) --> need to save this w for our ver of img-gen!!!
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy() # shape: (128, 128, 3)
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{this_outdir}/proj_'+target_fname.split('.')[0]+'.png')
            np.savez(f'{this_outdir}/projected_w_'+target_fname.split('.')[0]+'.npz', w=projected_w.unsqueeze(0).cpu().numpy())
            #print()
        

def get_key_from_item(item, dict_):
    keys = []
    for key_tmp in dict_:
        if dict_[key_tmp] == item:
            keys.append(key_tmp)
    
    return keys


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device,
    c                      # newly added by Chenqi
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    
    # newly added by Chenqi:
    label = torch.zeros([1, G.c_dim], device=device)
    label[:, c] = 1

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), label)  # [N, L, C]      # newly modified by Chenqi
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


#----------------------------------------------------------------------------

@click.command()
#@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default='results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds for generation', default='0-5')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--classes_orig', 'class_idxs', type=num_range, help='List of original Class labels', default='267,327')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir_root', help='Where to save the output images (root dir for generation!)', default='results/generate_inverInit/Birds_128-011200/val/', type=str, required=True, metavar='DIR')
#@click.option('--target_root', help='Root dir of Target image files to project to', default='datasets/Birds/', type=str, required=True, metavar='DIR')
@click.option('--map_pkl_dir', help='The location (full name) of corresponding cGAN_to_orig_cls_map_final.pkl', default='datasets/Birds_128_map/cGAN_to_orig_cls_map_final.pkl', type=str, required=True, metavar='DIR')
#@click.option('--num-steps',       help='Number of optimization steps', type=int, default=1000, show_default=True)
#@click.option('--seed',           help='Random seed for projection', type=int, default=303, show_default=True)
#@click.option('--run_projection',  help='Whether run the projection to get w files', type=bool, default=True, show_default=True)
@click.option('--outdir',          help='Where to save the output projected_w (Note: only do the projection&save once!)', default='results/project/Birds_128-011200/', type=str, required=True, metavar='DIR')

def my_generate_images_proj(
    #ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    class_idxs: Optional[List[int]], # List of original Class labels!
    noise_mode: str,
    outdir_root: str, # Where to save the output images (root dir WITHOUT specify cls folder!)
    map_pkl_dir: str,
    outdir: str # where we saved the projected_w npz files (generated in func run_projection_chenqi())
):
    """Generate images using pretrained network pickle and projected_w.
    """
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    
    # newly added by Chenqi:
    
    # load the cGAN_cls -> orig_clas map dict (both keys and items are int):
    f_pkl = open(map_pkl_dir,'rb')
    cGAN_to_orig_cls_map_final = pickle.load(f_pkl)
    f_pkl.close()
    
    
    # get the corresponding cGAN class label!
    for orig_cls in class_idxs:
        cGAN_cls = get_key_from_item(orig_cls, cGAN_to_orig_cls_map_final)
        class_idx = cGAN_cls[0]
        
        # assert the dir saving projected_w's corresponding to this single class exists! (using orig class label)
        proj_w_dir = outdir + str(orig_cls)
        assert(os.path.exists(proj_w_dir))
        
        # Note: shape of projected_w : torch.Size([12, 512])
        
        
        
        
    
    
    
    
    
    
    
    
    
    return





#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    #run_projection_chenqi() # pylint: disable=no-value-for-parameter
    
    my_generate_images_proj() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------




