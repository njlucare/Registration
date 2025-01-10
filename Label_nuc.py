#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tifffile as ti
import multiprocessing
from joblib import Parallel, delayed, Memory
from pystackreg import StackReg
from glob import glob
from tqdm import tqdm
from PAS_deconvolution2 import deconvolution_WSI
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.filters import threshold_local, threshold_otsu
from skimage.transform import resize as rsz
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform

class Nuc:
    def __init__(self,nuc,downsample=1):
        self.downsample = downsample
        self.nuc = ti.imread(nuc)[::self.downsample,::self.downsample]
        self.num_processes=multiprocessing.cpu_count()

    def register(self,mat,tf_mat,pad_dap=None,pad_hem=None,flip=None,rot=0,exp_factor=1):

        mat = zoom(mat,exp_factor,order=0)

        mat = np.rot90(mat,rot)

        if flip==1:
            mat = np.flipud(mat)
        elif flip==2:
            mat = np.fliplr(mat)

        if pad_dap is not None:
            pad_dap_h,pad_dap_w = pad_dap
            mat = self.__nuc_pad(mat, pad_dap_h, pad_dap_w)

        if pad_hem is not None:
            pad_hem_h,pad_hem_w = pad_hem

        tf_mat_normalized = tf_mat[0:2,:]
        offset = np.squeeze(tf_mat[2:,:])

        mask = affine_transform(
            mat,
            tf_mat_normalized,
            offset = offset,
            order=0,
            mode='constant',
            cval=0
            )

        return self.__hem_unpad(mask,pad_hem_h,pad_hem_w)

    def map_cells(self,clusters,centroids):

        assert centroids.shape[0] == clusters.shape[0]
        # assert centroids.shape[1] == 2, 'Must have two centroid columns'

        n_nuc = int(np.max(self.nuc))

        assert n_nuc == clusters.shape[0], 'Difference between image and matrix sizes'

        new_nucs = np.zeros_like(self.nuc)

        new_nucs = Parallel(n_jobs=self.num_processes,prefer="threads",require="sharedmem")(delayed(self.__map_slide)(i,centroids,clusters,self.nuc,new_nucs) for i in tqdm(range(n_nuc),desc='Nucleus #'))
        new_nucs = self.__map_cleanup(new_nucs)

        return new_nucs

    def get_tf_mat(self,hem_bin,flip=None,rot=0,exp_factor=1):

        transformations = {
            'AFFINE': StackReg.AFFINE
        }

        nuc_bin = (self.nuc > 0).astype('bool')

        nuc_bin = np.rot90(nuc_bin,rot)

        if flip==1:
            nuc_bin = np.flipud(nuc_bin)
        elif flip==2:
            nuc_bin = np.fliplr(nuc_bin)


        nuc_bin = zoom(nuc_bin,exp_factor,order=0)

        heightHisto, widthHisto = hem_bin.shape
        heightIF, widthIF = nuc_bin.shape

        pad_hem_h = heightIF-heightHisto if heightIF > heightHisto else 0
        pad_hem_w = widthIF-widthHisto if widthIF > widthHisto else 0

        pad_dap_h = heightHisto-heightIF if heightHisto > heightIF else 0
        pad_dap_w = widthHisto-widthIF if widthHisto > widthIF else 0

        #PAD THE IMAGES IF NEEDED

        hem_bin = (hem_bin > 0).astype('bool')

        nuc_bin=self.__nuc_pad(nuc_bin,pad_dap_h,pad_dap_w)
        hem_bin=self.__hem_pad(hem_bin,pad_hem_h,pad_hem_w)


        for i, (name, tf) in enumerate(transformations.items()):
            sr = StackReg(tf)
            #REGISTER!
            transformation_matrix = sr.register(hem_bin,nuc_bin)
            tf_mat_normalized = transformation_matrix[:2,:2]
            tf_mat_normalized = np.rot90(tf_mat_normalized,k=2)

            offset = transformation_matrix[:2,2]
            offset = np.flip(offset)
            offset = np.expand_dims(offset,0)

        tf_mat = np.concatenate((tf_mat_normalized,offset),axis=0)

        return tf_mat,(pad_dap_h,pad_dap_w),(pad_hem_h,pad_hem_w)


    def __map_slide(self,i,cluster_matrix,base_cluster,nucs,new_nucs):
        x_cent = int(cluster_matrix['PosY'][i]/self.downsample)
        y_cent = int(cluster_matrix['PosX'][i]/self.downsample)

        clus_0 = base_cluster[i]+1

        l = nucs[y_cent,x_cent]

        if l==0:
            return new_nucs
        else:
            new_nucs[nucs==l] = clus_0

            return new_nucs

    def __map_cleanup(self,new_nucs):
        if len(new_nucs)>0:
            new_nucs=new_nucs[0]
        else:
            new_nucs = []
        return new_nucs


    def __nuc_pad(self,nuc_bin,pad_dap_h,pad_dap_w):
        if (pad_dap_h % 2 != 0) & (pad_dap_w % 2 != 0):
            return np.pad(nuc_bin,((pad_dap_h//2+1,pad_dap_h//2),(pad_dap_w//2+1,pad_dap_w//2)),mode='constant')
        elif (pad_dap_h % 2 != 0) & (pad_dap_w % 2 == 0):
            return np.pad(nuc_bin,((pad_dap_h//2+1,pad_dap_h//2),(pad_dap_w//2,pad_dap_w//2)),mode='constant')
        elif (pad_dap_h % 2 == 0) & (pad_dap_w % 2 != 0):
            return np.pad(nuc_bin,((pad_dap_h//2,pad_dap_h//2),(pad_dap_w//2+1,pad_dap_w//2)),mode='constant')
        else:
            return np.pad(nuc_bin,((pad_dap_h//2,pad_dap_h//2),(pad_dap_w//2,pad_dap_w//2)),mode='constant')

    def __hem_pad(self,hem_bin,pad_hem_h,pad_hem_w):
        return np.pad(hem_bin,((pad_hem_h,0),(pad_hem_w,0)),mode='constant')

    def __hem_unpad(self,mask,pad_hem_h,pad_hem_w):
        return mask[pad_hem_h:,pad_hem_w:]
