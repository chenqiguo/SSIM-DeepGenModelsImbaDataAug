#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 18:55:00 2022

@author: guo.1648
"""

# Referenced from generate.py

# generate images for all needed classes in one run!

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

import pickle

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
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--classes_orig', 'class_idxs', type=num_range, help='List of original Class labels')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir_root', help='Where to save the output images (root dir!)', type=str, required=True, metavar='DIR')
@click.option('--map_pkl_dir', help='The location (full name) of corresponding cGAN_to_orig_cls_map_final.pkl', type=str, required=True, metavar='DIR')
def my_generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    class_idxs: Optional[List[int]], # List of original Class labels!
    noise_mode: str,
    outdir_root: str, # Where to save the output images (root dir WITHOUT specify cls folder!)
    map_pkl_dir: str
):
    """Generate images using pretrained network pickle.
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
        class_idx = cGAN_cls
        
        # get the output dir correspond to this single class! (using orig class label)
        outdir = outdir_root + '/' + str(orig_cls)
        
        os.makedirs(outdir, exist_ok=True)
        
        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected-w')
    
        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')
    
        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        
    



def get_key_from_item(item, dict_):
    keys = []
    for key_tmp in dict_:
        if dict_[key_tmp] == item:
            keys.append(key_tmp)
    
    return keys



#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    my_generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------













