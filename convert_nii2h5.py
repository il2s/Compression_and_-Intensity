from __future__ import print_function
import h5py
import nibabel as nib
import numpy as np
import os
import cv2
from scipy.ndimage.filters import gaussian_filter
import glob
from common import *

def load_nifti(filename, with_affine=False):
    img = nib.load(filename)
    data = img.get_data()
    data = np.squeeze(data)
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data

def extract_patch_save_to_file(split, dirs):
    #Extract patch and save in hdf5
    if split == 'train':
        save_path= train_path
    elif split =='val':
        save_path= val_path

    for index, folder_name in enumerate(dirs):

        file_T1_name = os.path.join(data_path,folder_name, 'pre', 'reg_T1.nii.gz')
        file_T1_IR_name = os.path.join(data_path,folder_name, 'pre', 'reg_IR.nii.gz')
        file_T2_FLAIR_name = os.path.join(data_path,folder_name, 'pre', 'FLAIR.nii.gz')
        label_file = os.path.join(data_path,folder_name,  'segm.nii.gz')

        img_T1,affine = load_nifti(file_T1_name, with_affine=True)
        img_T1=img_T1.astype(np.float32)

        img_T1_IR = load_nifti(file_T1_IR_name)
        img_T1_IR=img_T1_IR.astype(np.float32)

        img_T2_FLAIR = load_nifti(file_T2_FLAIR_name)
        img_T2_FLAIR=img_T2_FLAIR.astype(np.float32)

        label_img=load_nifti(label_file)
        label_img = label_img.astype(np.uint8)
        label_img [label_img > num_classes] = num_classes


        #===============Transpose from hxwxd to dxhxw==============
        img_T1 = img_T1.transpose(2, 0, 1)  # (3, 2, 0, 1)
        img_T1_IR = img_T1_IR.transpose(2, 0, 1)  # (3, 2, 0, 1)
        img_T2_FLAIR = img_T2_FLAIR.transpose(2, 0, 1)
        label_img=label_img.transpose(2, 0, 1)
        #===================End transpose==========================

        # Fix first slice IR of subject 14
        if (folder_name == '14'):
            img_T1_IR_slice = img_T1_IR[0, :, :]
            img_T1_IR_slice_next = img_T1_IR[1, :, :]
            img_T1_IR_slice[:, 0:20] = img_T1_IR_slice_next[:, 0:20]
            img_T1_IR[0, :, :] = img_T1_IR_slice

        #======================normalize to 0 mean and 1 variance==
        img_T1 = (img_T1 - img_T1.mean()) / img_T1.std()
        img_T1_IR = (img_T1_IR - img_T1_IR.mean()) / img_T1_IR.std()
        img_T2_FLAIR = (img_T2_FLAIR - img_T2_FLAIR.mean()) / img_T2_FLAIR.std()

        print (img_T1.shape, img_T1.shape, img_T2_FLAIR.shape)
        print (label_img.min(), label_img.max())
        print (img_T1.min(),img_T1.max())
        print(img_T1_IR.min(), img_T1_IR.max())
        print(img_T2_FLAIR.min(), img_T2_FLAIR.max())

        #====Expand to 5D blob:  one more dimension about #sample and channel===============
        img_T1 = img_T1[None, None, :, :, :]
        img_T1_IR = img_T1_IR[None, None, :, :, :]
        img_T2_FLAIR = img_T2_FLAIR[None, None, :, :, :]
        label_img  = label_img[None, :, :, :]
        #=======End expand=================================

        data_multimodal = np.concatenate((img_T1, img_T1_IR, img_T2_FLAIR), 1)  # Ignore T1_IR
        print (data_multimodal.shape, label_img.shape)
        with h5py.File(save_path + 'train_%s.h5' % str(folder_name), 'w') as f:
            f['data'] = data_multimodal
            f['label'] = label_img



data_path = './training'
train_path = './data_train/'
val_path = './data_val/'

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)
_, dirs, _ = os.walk(data_path).__next__()

# Select 148 as validation
dirs.remove("148")
dirs_train = dirs
dirs_val = ['148']


extract_patch_save_to_file('train', dirs_train)
extract_patch_save_to_file('val', dirs_val)

