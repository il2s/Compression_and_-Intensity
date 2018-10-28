import h5py
import torch
import torch.utils.data as data
import glob
import os
from common import *
import numpy as np
import nibabel as nib
#BCHW order
class H5Dataset(data.Dataset):

    def __init__(self, root_path, crop_size=crop_size, mode='train', check_invalid=False):
        self.hdf5_list = [x for x in glob.glob(os.path.join(root_path, '*.h5'))]
        self.crop_size = crop_size
        self.mode = mode
        self.check_invalid=check_invalid
        if (self.mode == 'train'):
            self.hdf5_list =self.hdf5_list + self.hdf5_list

        self.data_lst  = np.zeros((len(self.hdf5_list), 1, data_dim, xdim, ydim, zdim), dtype=np.float32)
        self.label_lst = np.zeros((len(self.hdf5_list), 1, xdim, ydim, zdim), dtype=np.uint8)
        for index in range (len(self.hdf5_list)):
            h5_file = h5py.File(self.hdf5_list[index])
            self.data_lst[index,...] = h5_file.get('data')
            self.label_lst[index,...]= h5_file.get('label')
            h5_file.close()

    def __getitem__(self, index):
        self.data = self.data_lst[index,...]
        self.label = self.label_lst[index,...]
        #------Random crop------------------
        _, _, C, H, W = self.data.shape
        # print("c :",C, "H : ",H,"W:",W)
        if (self.mode=='train'):
            cx = random.randint(0, C - self.crop_size[0])
            cy = random.randint(0, H - self.crop_size[1])
            cz = random.randint(0, W - self.crop_size[2])

        elif (self.mode == 'val'):
            # -------Center crop----------
            cx = (C - self.crop_size[0])//2
            cy = (H - self.crop_size[1])//2
            cz = (W - self.crop_size[2])//2

        # Check invalid region: less object, larger background
        if (self.check_invalid):
            invalid_label=self.label[:, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]
            while (invalid_label.sum()<1):
                #print('---------Invalid region-----------:', invalid_label.sum())
                cx = random.randint(0, C - self.crop_size[0])
                cy = random.randint(0, H - self.crop_size[1])
                cz = random.randint(0, W - self.crop_size[2])
                invalid_label = self.label[:, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1],
                                cz: cz + self.crop_size[2]]

        self.data_crop  = self.data [:, :, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]
        self.label_crop = self.label[:, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]

        # ------End random crop-------------
        #h5_file.close()
        return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                torch.from_numpy(self.label_crop[0,:,:,:]).long())

    def __len__(self):
        return len(self.hdf5_list)
