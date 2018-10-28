from common import *
import torch.utils.data as dataloader
from dataloader2D import H5Dataset
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from segmentor import DenseNet #v3 best
from torch.autograd import Variable
from write_torch2nii import write_predict2nii, write_label2nii,write_torch2nii
from metrics import dice
import numpy as np
import torch.nn.functional as F
import nibabel as nib
import  time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import shutil

from segmentor import *
# from upsample import *

# labels for adversarial training
pred_label = 0
gt_label = 1
label_smooth=0.15
pred_smooth=0.0
import visdom
viz = visdom.Visdom()
def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title
            #legend=_legend
        )
    )


def update_vis_plot(epoch, loss, window1, update_type):
    viz.line(
        X=torch.ones((1)).cpu() * epoch,
        Y=loss.unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )

def update_vis_dsc_plot(epoch, dsc, window1, update_type):
    viz.line(
        X=torch.ones((1)).cpu() * epoch,
        Y=dsc.unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )

def save_checkpoint(state, is_best, filename= './checkpoints/model_best_adv.pth.tar'):
    #torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------init Seg and init D---------------
# model_S = DenseNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=9,drop_rate=0.2).to(device)
# model_S = inceptionv4(pretrained=True).to(device)
model_S = unet().to(device)
model_name = 'best_unet(08)'
train_folder = './data_train'
val_folder = './data_val'

# --------------Loss---------------------------
criterion_S = nn.CrossEntropyLoss(ignore_index=ignore_label).to(device)
#criterion_L1 = nn.L1Loss().to(device)

best_dsc=0
best_epoch=0
dsc_max=0
epoch_max=0
# --------------Start Training and Validation ---------------------------
if __name__ == '__main__':
    #-----------------Check command--------------------------------------
    parser = argparse.ArgumentParser(description="3D-SkipDenseSeg")
    parser.add_argument('--Adv', type=int, default=0, help='Adversarial Training mode')
    parser.add_argument('--resume', type=int, default=0, help='Resume')
    args = parser.parse_args()


    num_epoch_S=num_epoch
    pre_trained = False
    check_invalid =False

    # checkpoint_name = checkpoint_name_noAdv
    print('-----------------Start Generator Training--------------------------')
    print('Rate     | epoch  | Loss seg| DSC_val')
    title ='Without Adversarial Traning'

    # setup optimizer
    # optimizer_S = optim.SGD(model_S.parameters(), lr=1e-2, momentum=momentum_S, weight_decay=1e-4)
    optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S_Adam, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.1)

    #-----------------------Training--------------------------------------
    mri_data_train = H5Dataset(train_folder, mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True, num_workers=num_workers)
    mri_data_val = H5Dataset(val_folder, mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)

    epoch_seg_plot = create_vis_plot('Epoch', 'Loss', title, 'Training Generator loss')
    epoch_DSC_plot = create_vis_plot('Epoch', 'DSC', title, 'Dice')

    count_t = time.time()

    for epoch in range(best_epoch, num_epoch_S):
        scheduler_S.step(epoch)
        model_S.train()
        for i, data in enumerate(trainloader):
            images, targets = data
            if (images.size(0)==2):
                continue
            # Set mode cuda if it is enable, otherwise mode CPU
            images = images.to(device)
            targets = targets.to(device)

            # if torch.sum(targets) < 10:
            #     continue
            # ###########################
            # #Augmentation data
            # ##########################
            # if random.random() > 0.8:
            #     images = images.transpose(3, 4)
            #     targets = targets.transpose(2, 3)

            ############################
            # (1) Update G network: maximize log(D(S(x)))
            ###########################
            optimizer_S.zero_grad()
            # z-->S(z)
            # outputs,outputs1,outputs2,outputs3,outputs4  = model_S(images)
            outputs  = model_S(images)

            # ---Crossentropy loss for Seg----------
            loss_seg0 = criterion_S(outputs, targets)
            # loss_seg1 = criterion_S(outputs1, targets)
            # loss_seg2 = criterion_S(outputs2, targets)
            # loss_seg3 = criterion_S(outputs3, targets)
            # loss_seg4 = criterion_S(outputs4, targets)
            # ---------------------------------------
            loss_seg = loss_seg0 #+ 0.1 * (loss_seg1 + loss_seg2 + loss_seg3 + loss_seg4)

            loss_seg.backward()
            optimizer_S.step()  # Only optimize S parameters

        with torch.no_grad():
            model_S.eval()
            for data_val in valloader:
                images_val, targets_val = data_val
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)

                outputs_val = model_S(images_val)
                _, predicted = torch.max(outputs_val.data, 1)

                # ----------Compute dice-----------
                predicted_val = predicted.data.cpu().numpy()
                targets_val = targets_val.data.cpu().numpy()

                dsc = []
                for i in range(1, num_classes):  # ignore Background 0
                    dsc_i = dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)

        # -------------------Debug-------------------------
        for param_group in optimizer_S.param_groups:
            check = time.time() - count_t
            c_h = math.trunc(float(check)) // 3600
            c_m = math.trunc(float(check) - c_h * 3600) // 60
            c_s = math.trunc(float(check)) % 60
            c_c = str(c_h) + 'h ' + str(c_m) + 'm ' + str(c_s) + 's'
            if (dsc >= dsc_max) and ((epoch+1 < 11) or (epoch+1 > 1000)):
                dsc_max = dsc
                epoch_max = epoch+1
                # if epoch > 699:
                torch.save(model_S.state_dict(), './checkpoints/' + '%s_%s.pth' % (str(epoch).zfill(5),
                                                                                   model_name + '(' + c_c + ')'))
                print('%0.6f | %6d | %s | %0.5f | %0.5f | %0.5f | %d | %d' % ( \
                    param_group['lr'], epoch+1, c_c,
                    loss_seg.data.cpu().numpy(),
                    dsc, dsc_max, epoch_max, targets.size(0)), end='\r')
            else:
                print('%0.6f | %6d | %s | %0.5f | %0.5f | %0.5f | %d | %d' % ( \
                    param_group['lr'], epoch + 1, c_c,
                    loss_seg.data.cpu().numpy(),
                    dsc, dsc_max, epoch_max, targets.size(0)), end='\r')

        # Save checkpoint
        if (epoch % 100) == 0 or epoch == (num_epoch - 1):
            torch.save(model_S.state_dict(), './checkpoints/' + '%s_%s.pth' % (str(epoch).zfill(5),
                                                                               model_name + '(' + c_c + ')'))
        # if epoch % 1000 == (1000-1) or epoch==(num_epoch-1):
        #     torch.save(model_S.state_dict(), './checkpoints/' + '%s_%s.pth' % (str(epoch+1).zfill(5), checkpoint_name))

        # is_best = dsc > best_dsc
        # best_dsc = max(dsc, best_dsc)
        # if (is_best):
        #     best_epoch = epoch + 1
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': '3D_DenseSeg_GAN',
        #     'state_dict': model_S.state_dict(),
        #     'best_prec1': best_dsc,
        #     'optimizer': optimizer_S.state_dict(),
        # }, is_best, filename= './checkpoints/model_best.pth.tar')


        # for tag, value in info.items():
        #    logger.scalar_summary(tag, value, epoch + 1)
        update_vis_plot(epoch+1, loss_seg, epoch_seg_plot, 'append')
        update_vis_plot(epoch+1, torch.from_numpy(np.array(dsc)).float(), epoch_DSC_plot, 'append')
        #loss_seg


    print ("\nFinished Training with best dsc: ", best_dsc , "at epoch: ", best_epoch)
