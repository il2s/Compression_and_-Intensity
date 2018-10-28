import torch.backends.cudnn as cudnn
from write_torch2nii import write_wholepred2nii,write_wholelabel2nii
from metrics import dice
import numpy as np
import torch
import random
import glob
import h5py
import os
import time
import operator
from common import *

from segmentor import *

def search(dirname):
    filenames = os.listdir(dirname)
    names = []
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.pth':
           names.append( full_filename.split("/")[-1])
           #print(filename)
    return names


# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TarnausNet16(pretrained=True).to(device)
# model = inceptionv4(pretrained=True).to(device)
# model = resnext101_64x4d(pretrained=True).to(device)
model = unet().to(device)

layer = "until 4"

# ----------------------- read all file --------------------------------

# ------------TODO: Move it to other file--------------
# --------------Start Training and Testing ---------------------------
if __name__ == '__main__':
    # ------------------------ load files --------------------------
    filenames = search("/home/asi/machine2/MRBrainS18/checkpoints")
    print("number of files : ", len(filenames))

    filenames.sort()
    dict = {}
    print('Data       | Note   | class1| class2|class3|class4|class5|class6|class7|class8| Avg.|')

    for filename in filenames:
        m_name = filename
        # -----------------------Testing-------------------------------------
        # -----------------------Load the checkpoint (weights)---------------
        checkpoint ='./checkpoints/' + filename
        state_dict =  torch.load(checkpoint)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in state_dict and param.size() == state_dict[name].size():
                new_params[name].copy_(state_dict[name])
        model.load_state_dict(new_params)
        model.eval()
        # -----------------------Load testing data----------------------------
        test_path='./data_val'

        index_file = 0
        hdf5_list = [x for x in glob.glob(os.path.join(test_path, '*.h5'))]
        h5_file = h5py.File(hdf5_list[index_file])
        # print ('Prediction for ', hdf5_list[index_file])
        image = h5_file.get('data')
        label = h5_file.get('label')

        _, _, C, H, W = image.shape
        whole_pred = np.zeros((1,)+(num_classes,) + image.shape[2:])
        count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5

        # no update parameter gradients during testing
        with torch.no_grad():
            for i in range(C):
                deep = i
                image_crop = image[:, :, deep , : , :]
                image_crop = torch.from_numpy(image_crop[:, :, :, :]).float()
                #print (image_crop.shape)
                #------------------Original crop-------------------------------
                image_crop_none = image_crop
                image_crop_none = image_crop_none.to(device)
                outputs = model(image_crop_none)

                whole_pred[slice(None), slice(None), deep, :, :] = outputs

            for t in range(len(whole_pred)):
                whole_pred = whole_pred[0, :, :, :, :]
                # print("whole_pred : ", whole_pred.shape)
                whole_pred = np.argmax(whole_pred, axis=0)
                label = label[0, :, :, :]
                # print("whole_label : ", label.shape)
                write_wholepred2nii (whole_pred, filename)
                write_wholelabel2nii(label)
                # #----------Compute dice-----------
                dsc = []
                # print ('-------------------------')

                for i in range(1, num_classes):
                    dsc_i = dice(whole_pred, label, i)
                    dsc_i=round(dsc_i*100,2)
                    dsc.append(dsc_i)
                #print ('-------------------------')
                datetime= time.strftime("%d/%m/%Y")

                # print(" =================== ", t, " slide = ==================================")
                print('%s | %s | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f| %2.2f | %2.2f | %2.2f | %2.2f | %s' % ( \
                            datetime,
                            m_name,
                            dsc[0],
                            dsc[1],
                            dsc[2],
                            dsc[3],
                            dsc[4],
                            dsc[5],
                            dsc[6],
                            dsc[7],
                            np.mean(dsc),
                            layer))

                dict[filename] = np.mean(dsc)
    print("==============================================================================")
    print(sorted(dict.items(), key=operator.itemgetter(1), reverse=True))