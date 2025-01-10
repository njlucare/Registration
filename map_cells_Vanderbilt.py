#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tifffile as ti
from Label_nuc import Nuc
from glob import glob
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform


slide_dir = '/orange/pinaki.sarder/nlucarelli/Vanderbilt/'
mapping = '/orange/pinaki.sarder/nlucarelli/Vanderbilt/data/mapping.xlsx'
downsample = 2

slide_info = pd.read_excel(mapping)

slides = list(slide_info['nuc'])

idx = 0

for slide in tqdm(slides):

    output_name = slide.rsplit('/',1)[0] + '/'+slide.split('/')[-1].split('.tif')[0] + '_Registered.tif'

    if os.path.exists(output_name):
        print(f'Already mapped, skipping: {output_name}')
        idx+=1
        continue

    obj = Nuc(slide,downsample=2)

    clusters = slide_info.loc[slide_info['nuc'] == slide, 'clusters'][idx]
    centroids = slide_info.loc[slide_info['nuc'] == slide, 'centroids'][idx]
    slide_labels = pd.read_csv(slide_info.loc[slide_info['nuc'] == slide, 'slide_labels'][idx])['x'].to_numpy()
    slide_label = slide_info.loc[slide_info['nuc'] == slide, 'slide_label'][idx]

    flip = slide_info.loc[slide_info['nuc'] == slide, 'flip'][idx]
    rot = slide_info.loc[slide_info['nuc'] == slide, 'rot'][idx]
    exp_factor = slide_info.loc[slide_info['nuc'] == slide, 'exp_factor'][idx]

    clusters = pd.read_csv(clusters)['communities'][slide_labels==slide_label].to_numpy()
    centroids = pd.read_csv(centroids)

    mapped_cells = obj.map_cells(clusters, centroids)
    ti.imwrite(slide.rsplit('/',1)[0] + '/' + slide.split('/')[-1].split('.tif')[0] + '_overlay.tif',mapped_cells,photometric='minisblack')
    # mapped_cells = ti.imread(slide.rsplit('/',1)[0] + '/' + slide.split('/')[-1].split('.tif')[0] + '_overlay.tif')#[::downsample,::downsample]
    he = ti.imread(slide_info.loc[slide_info['nuc'] == slide, 'he'][idx])[::downsample,::downsample]

    tf_mat,pad_dap,pad_hem = obj.get_tf_mat(he,flip,rot,exp_factor)

    registered = obj.register(mapped_cells,tf_mat,pad_dap=pad_dap,pad_hem=pad_hem,flip=flip,rot=rot,exp_factor=exp_factor)

    ti.imwrite(slide.rsplit('/',1)[0] + '/'+slide.split('/')[-1].split('.tif')[0] + '_Registered.tif',registered,photometric='minisblack')
    del registered, mapped_cells
    idx+=1
