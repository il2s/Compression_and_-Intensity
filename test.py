from segmentor import DenseNet
import torch.backends.cudnn as cudnn
from write_torch2nii import write_wholepred2nii,write_wholelabel2nii,write_prob2nii
from metrics import dice
import numpy as np
import torch
import random
import glob
import h5py
import os
import time
from common import *
import argparse
import nibabel as nib
import torch.nn.functional as F
def load_nifti(filename, with_affine=False):
    img = nib.load(filename)
    data = img.get_data()
    data = np.squeeze(data)
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data

outputs_path='./outputs/148'
if not os.path.exists(outputs_path):
    os.makedirs(outputs_path)

# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DenseNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=9).to(device)
# ------------TODO: Move it to other file--------------
# --------------Start Training and Testing ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D-SkipDenseSeg")
    parser.add_argument("--iters", type=int, default=10000,
                        help="checkpoints number")
    parser.add_argument('--Adv', type=int, default=0, help='Adversarial Training mode')
    #args = parser.parse_args()

    parser.add_argument('--best', type=int, default=0, help='best checkpoint')
    args = parser.parse_args()
    if args.Adv==1:
        # -----------Adversarial Training---------------------------------
        checkpoint_name = checkpoint_name_Adv
    else:
        checkpoint_name = checkpoint_name_noAdv


    checkpoint = './checkpoints/' + str(args.iters).zfill(5) + '_' + checkpoint_name + '.pth'



    note = str(args.iters) + '_' + checkpoint_name + '.pth'
    # -----------------------Testing-------------------------------------
    # -----------------------Load the checkpoint (weights)---------------

    if args.best == 1:
        checkpoint = './checkpoints/model_best_adv.pth.tar'
        checkpoint_dict = torch.load(checkpoint)
        best_epoch = checkpoint_dict['epoch']
        best_dsc = checkpoint_dict['best_prec1']
        net.load_state_dict(checkpoint_dict['state_dict'])
        print ('best dsc ', best_dsc, ' best epoch ', best_epoch)
    else:
        saved_state_dict = torch.load(checkpoint)
        net.load_state_dict(saved_state_dict)
    print('Checkpoint: ', checkpoint)

    net.eval()
    # -----------------------Load testing data----------------------------

    index_file = 0
    xstep = 1
    ystep = 8 # 16
    zstep = 8  # 16
    image_flip_dims=[4,3]
    label_flip_dims=[3,2]

    data_path = './training'
    root, dirs, files = os.walk(data_path).__next__()

    for index, folder_name in enumerate(dirs):
        if folder_name != '148':
            continue

        file_T1_name = os.path.join(data_path, folder_name, 'pre', 'reg_T1.nii.gz')
        file_T1_IR_name = os.path.join(data_path, folder_name, 'pre', 'reg_IR.nii.gz')
        file_T2_FLAIR_name = os.path.join(data_path, folder_name, 'pre', 'FLAIR.nii.gz')
        label_file = os.path.join(data_path, folder_name, 'segm.nii.gz')

        img_T1, affine = load_nifti(file_T1_name, with_affine=True)
        img_T1 = img_T1.astype(np.float32)

        img_T1_IR = load_nifti(file_T1_IR_name)
        img_T1_IR = img_T1_IR.astype(np.float32)

        img_T2_FLAIR = load_nifti(file_T2_FLAIR_name)
        img_T2_FLAIR = img_T2_FLAIR.astype(np.float32)

        label_img = load_nifti(label_file)
        label_img = label_img.astype(np.uint8)
        label_img[label_img > num_classes] = num_classes

        # ===============Transpose from hxwxd to dxhxw==============
        img_T1 = img_T1.transpose(2, 0, 1)  # (3, 2, 0, 1)
        img_T1_IR = img_T1_IR.transpose(2, 0, 1)  # (3, 2, 0, 1)
        img_T2_FLAIR = img_T2_FLAIR.transpose(2, 0, 1)
        label_img = label_img.transpose(2, 0, 1)
        # ===================End transpose==========================

        # Fix first slice IR of subject 14
        if (folder_name == '14'):
            img_T1_IR_slice = img_T1_IR[0, :, :]
            img_T1_IR_slice_next = img_T1_IR[1, :, :]
            img_T1_IR_slice[:, 0:20] = img_T1_IR_slice_next[:, 0:20]
            img_T1_IR[0, :, :] = img_T1_IR_slice

        # ======================normalize to 0 mean and 1 variance==
        img_T1 = (img_T1 - img_T1.mean()) / img_T1.std()
        img_T1_IR = (img_T1_IR - img_T1_IR.mean()) / img_T1_IR.std()
        img_T2_FLAIR = (img_T2_FLAIR - img_T2_FLAIR.mean()) / img_T2_FLAIR.std()

        # ====Expand to 5D blob:  one more dimension about #sample and channel===============
        img_T1 = img_T1[None, None, :, :, :]
        img_T1_IR = img_T1_IR[None, None, :, :, :]
        img_T2_FLAIR = img_T2_FLAIR[None, None, :, :, :]
        label_img = label_img[None, :, :, :]
        # =======End expand=================================

        image = np.concatenate((img_T1, img_T1_IR, img_T2_FLAIR), 1)  # Ignore T1_IR
        label = label_img

        print (image.shape, label.shape)

        _, _, C, H, W = image.shape
        deep_slices   = np.arange(0, C - crop_size[0] + xstep, xstep)
        height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
        width_slices  = np.arange(0, W - crop_size[2] + zstep, zstep)

        whole_pred = np.zeros((1,)+(num_classes,) + image.shape[2:])
        count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5

        # no update parameter gradients during testing
        with torch.no_grad():
            for i in range(len(deep_slices)):
                for j in range(len(height_slices)):
                    for k in range(len(width_slices)):
                        deep = deep_slices[i]
                        height = height_slices[j]
                        width = width_slices[k]
                        image_crop = image[:, :, deep   : deep   + crop_size[0],
                                                 height : height + crop_size[1],
                                                 width  : width  + crop_size[2]]

                        image_crop = torch.from_numpy(image_crop[:, :, :, :, :]).float().to(device)
                        #------------------Original crop-------------------------------
                        image_crop_none = image_crop
                        #image_crop_none = image_crop_none.to(device)
                        outputs = net(image_crop_none)
                        # Average
                        whole_pred[slice(None), slice(None), deep: deep + crop_size[0],
                                    height: height + crop_size[1],
                                    width: width + crop_size[2]] += outputs.data.cpu().numpy()

                        count_used[deep: deep + crop_size[0],
                                    height: height + crop_size[1],
                                    width: width + crop_size[2]] += 1

                        # #-------------Testing time augmentation------------------------
                        for dim_ind in range (len(image_flip_dims)):
                            #image_crop_aug = image_crop_aug.to(device)
                            image_crop_copy= image_crop.clone()
                            image_crop_aug = image_crop.flip(image_flip_dims[dim_ind])
                            outputs = net(image_crop_aug)
                            outputs = outputs.flip(image_flip_dims[dim_ind])

                            # ----------------Major voting-------------------------------
                            # _, temp_predict = torch.max(outputs.data, 1)
                            # for labelInd in range(4):  # note, start from 0
                            #     currLabelMat = np.where(temp_predict == labelInd, 1, 0)  # true, vote for 1, otherwise 0
                            #     whole_pred[slice(None), labelInd, deep: deep + crop_size[0],
                            #     height: height + crop_size[1],
                            #     width: width + crop_size[2]] += currLabelMat

                            # ---------------------Average---------------------------------
                            whole_pred[slice(None), slice(None),  deep   : deep   + crop_size[0],
                                                                 height : height + crop_size[1],
                                                                 width  : width  + crop_size[2]] += outputs.data.cpu().numpy()


                            count_used[deep   : deep   + crop_size[0],
                                       height : height + crop_size[1],
                                       width  : width  + crop_size[2]] += 1

                            ##image_crop_aug = image_crop_copy.transpose(3, 4)
                           # #outputs = net(image_crop_aug)
                           # outputs = outputs.transpose(3, 4)

                            # ----------------Major voting-------------------------------
                            # _, temp_predict = torch.max(outputs.data, 1)
                            # for labelInd in range(4):  # note, start from 0
                            #     currLabelMat = np.where(temp_predict == labelInd, 1, 0)  # true, vote for 1, otherwise 0
                            #     whole_pred[slice(None), labelInd, deep: deep + crop_size[0],
                            #     height: height + crop_size[1],
                            #     width: width + crop_size[2]] += currLabelMat

                            # ---------------------Average---------------------------------
                            #whole_pred[slice(None), slice(None), deep: deep + crop_size[0],
                            #height: height + crop_size[1],
                           # width: width + crop_size[2]] += F.softmax(outputs, dim=1).data.cpu().numpy()

                           # count_used[deep: deep + crop_size[0],
                            #height: height + crop_size[1],
                           # width: width + crop_size[2]] += 1

            whole_pred = whole_pred / count_used

            # whole_pred_tensor= torch.sigmoid(torch.from_numpy(whole_pred))
            # whole_pred = whole_pred_tensor.data.cpu().numpy()

            whole_pred = whole_pred[0, :, :, :, :]
            #whole_pred[whole_pred<0]=0
            for i in range(1, num_classes):
                write_prob2nii(whole_pred[i,...], i)
                print (np.min(whole_pred[i,...]),np.max(whole_pred[i,...]))
            whole_pred = np.argmax(whole_pred, axis=0)
            write_wholepred2nii (whole_pred)
            label = label[0, :, :, :]
            write_wholelabel2nii(label)
            # #----------Compute dice-----------
            dsc = []
            print ('-----------9 classes-----------')
            for i in range(1, num_classes):
                dsc_i = dice(whole_pred, label, i)
                dsc_i=round(dsc_i*100,2)
                dsc.append(dsc_i)
            #print ('-------------------------')
            datetime= time.strftime("%d/%m/%Y")
            print('Data       | Note   | class1| class2|class3|class4|class5|class6|class7|class8| Avg.|')
            print('%s | %s | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f| %2.2f | %2.2f | %2.2f | %2.2f |' % ( \
                        datetime,
                        note,
                        dsc[0],
                        dsc[1],
                        dsc[2],
                        dsc[3],
                        dsc[4],
                        dsc[5],
                        dsc[6],
                        dsc[7],
                        np.mean(dsc)))

            print ('-----------4 classes-----------')
            whole_pred = convert2fourclass(whole_pred)
            label = convert2fourclass(label)
            dsc=[]
            for i in range(1, 4):
                dsc_i = dice(whole_pred, label, i)
                dsc_i=round(dsc_i*100,2)
                dsc.append(dsc_i)
            datetime= time.strftime("%d/%m/%Y")
            print('Data       | Note   | class1| class2|class3| Avg.|')
            print('%s | %s | %2.2f | %2.2f | %2.2f | %2.2f |' % ( \
                        datetime,
                        note,
                        dsc[0],
                        dsc[1],
                        dsc[2],
                        np.mean(dsc)))
