import h5py
import torch
import torch.utils.data as data
import glob
import os
import random
import numpy as np
import nibabel as nib
#BCHW order

def load_nifti(filename, with_affine=False):
    img = nib.load(filename)
    data = img.get_data()
    data = np.squeeze(data)
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data
def write_torch2nii(image, index):
    scalar_img, affine = load_nifti('./training/148/pre/reg_T1.nii.gz', with_affine=True)
    img_np=image.data.cpu().numpy()
    #Save back to nii
    #Transpose to HxWxC
    img_np=img_np.transpose(0,3,4,2,1)# BCDHW->BHWDC
    # print(img_np.shape)
    for j in range(img_np.shape[0]):
        nib.save(nib.Nifti1Image(img_np[j,:].astype(np.float32), affine), './outputs/image_conv_crop_%d_%d.nii' %(index,j))

def write_predict2nii(image):
    scalar_img, affine = load_nifti('./training/148/pre/reg_T1.nii.gz', with_affine=True)
    img_np=image.data.cpu().numpy()
    #Save back to nii
    #Transpose to HxWxC
    img_np=img_np.transpose(0,2,3,1)# BDHW->BHWDC

    img_np=img_np[0,:,:,:]
    # print(img_np.shape)
    nib.save(nib.Nifti1Image(img_np.astype(np.uint8), affine), './outputs/pred_label.nii.gz')

def write_label2nii(image):
    scalar_img, affine = load_nifti('./training/148/pre/reg_T1.nii.gz', with_affine=True)
    img_np=image.data.cpu().numpy()
    #Save back to nii
    #Transpose to HxWxC
    img_np=img_np.transpose(0,2,3,1)# BDHW->BHWDC

    img_np=img_np[0,:,:,:]
    # print(img_np.shape)
    nib.save(nib.Nifti1Image(img_np.astype(np.uint8), affine), './outputs/true_label.nii')

def write_wholepred2nii(image,filename):
    scalar_img, affine = load_nifti('./training/148/pre/reg_T1.nii.gz', with_affine=True)
    image=image.transpose(1,2,0)
    # print(image.shape)
    nib.save(nib.Nifti1Image(image.astype(np.uint8), affine), './outputs/148/'+filename+'.nii.gz')

def write_wholelabel2nii(image):
    scalar_img, affine = load_nifti('./training/148/pre/reg_T1.nii.gz', with_affine=True)
    image = image.transpose(1, 2, 0)
    nib.save(nib.Nifti1Image(image.astype(np.uint8), affine), './outputs/148/whole_label.nii.gz')

def write_prob2nii(image, i):
    scalar_img, affine = load_nifti('./training/148/pre/reg_T1.nii.gz', with_affine=True)
    image = image.transpose(1, 2, 0)
    nib.save(nib.Nifti1Image(image.astype(np.float32), affine), './outputs/148/prob_class%d.nii.gz'% i)