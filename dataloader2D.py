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
        if (self.mode == 'train'):
            self.hdf5_list =self.hdf5_list + self.hdf5_list# + self.hdf5_list + self.hdf5_list
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
        # print ("dddd", self.data.shape,self.label.shape)
        if (self.mode=='train'):
            rnd_slice = random.randint(0, C-1)
        elif (self.mode == 'val'):
            # -------Center slcie----------
            rnd_slice = C//2-1

        self.data_slice  = self.data [:, :, rnd_slice, :, :]
        self.label_slice = self.label[:, rnd_slice, :, :]
        while (self.label_slice.sum()==0):
            rnd_slice = random.randint(0, C - 1)
            self.data_slice = self.data[:, :, rnd_slice, :, :]
            self.label_slice = self.label[:, rnd_slice, :, :]


        #print (self.data_slice.shape, self.label_slice.shape)
        # ------End random crop-------------
        return (torch.from_numpy(self.data_slice[0,:,:,:]).float(),
                torch.from_numpy(self.label_slice[0,:,:]).long())

    def __len__(self):
        return len(self.hdf5_list)

# import torch.utils.data as dataloader
# from dataloader2D import H5Dataset
# import torchvision.utils as v_u
# mri_data_train = H5Dataset("./data_train", mode='train')
# trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True, num_workers=2)
# for i, data in enumerate(trainloader):
#     images, targets = data
#     print (images.size(),targets.size())
#     v_u.save_image(images[0,:,:,:], 'image1.jpg')
