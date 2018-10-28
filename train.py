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
from discriminator import FCDiscriminator
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

model_D = FCDiscriminator(num_classes=num_classes).to(device)
# --------------Loss---------------------------
criterion_S = nn.CrossEntropyLoss(ignore_index=ignore_label).to(device)
criterion_D = nn.BCEWithLogitsLoss().to(device)
#criterion_L1 = nn.L1Loss().to(device)

model_D.train()
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

    if args.Adv==1:
        #-----------Adversarial Training---------------------------------
        num_epoch_S=0
        num_epoch_S_G=20000
        pre_trained = True
        check_invalid= True
        step_size_S=4000
        step_size_D=4000
        checkpoint_name = checkpoint_name_Adv
        print('-----------------Start Adversarial Training--------------------------')
        print('Rate     | epoch  | Loss seg| Loss adv| loss_D  | Loss total| DSC_val  | DSC_max  | epoch_max')
        title ='With Adversarial Traning'
        epoch_adv_plot = create_vis_plot('Epoch', 'Loss', title, 'Training Adv loss')
        epoch_D_plot = create_vis_plot('Epoch', 'Loss', title, 'Training D loss')
    else:
        num_epoch_S=num_epoch
        num_epoch_S_G=0
        pre_trained = False
        check_invalid =False

        # checkpoint_name = checkpoint_name_noAdv
        print('-----------------Start Generator Training--------------------------')
        print('Rate     | epoch  | Loss seg| DSC_val')
        title ='Without Adversarial Traning'

    # setup optimizer
    # optimizer_S = optim.SGD(model_S.parameters(), lr=1e-2, momentum=momentum_S, weight_decay=1e-4)
    optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S_Adam, weight_decay=1e-4, betas=(0.9, 0.999))
    # optimizer_D = optim.SGD(model_D.parameters(), lr=lr_D, momentum=momentum_S)
    optimizer_D = optim.Adam(model_D.parameters(), lr=lr_D_Adam, betas=(0.95, 0.999))
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=step_size_D, gamma=0.1)

    #-----------------------Training--------------------------------------
    mri_data_train = H5Dataset(train_folder, mode='train')
    # mri_data_train = H5Dataset("./each_data_train/T1-IR", mode='train', check_invalid=check_invalid)
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True, num_workers=num_workers)
    mri_data_val = H5Dataset(val_folder, mode='val')
    # mri_data_val = H5Dataset("./each_data_val/T1-IR", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)

    if (pre_trained):
        pre_trained_iters=15000
        checkpoint_pretrained = './checkpoints/' + str(pre_trained_iters).zfill(5) + '_' + checkpoint_name_noAdv + '.pth'
        saved_state_dict = torch.load(checkpoint_pretrained)
        model_S.load_state_dict(saved_state_dict)
    if(args.resume==1):
        checkpoint = './checkpoints/model_best.pth.tar'
        checkpoint_dict = torch.load(checkpoint)
        best_epoch = checkpoint_dict['epoch']
        best_dsc = checkpoint_dict['best_prec1']
        model_S.load_state_dict(checkpoint_dict['state_dict'])
        optimizer_S.load_state_dict(checkpoint_dict['optimizer'])
        scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=best_epoch, gamma=0.1)

        print ('best dsc ', best_dsc, ' best epoch ', best_epoch)
    ############################
    # Train G a while
    ###########################

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


    for epoch in range (num_epoch_S_G):
        #-----------Adjust Learning rate S and D-----
        scheduler_S.step(epoch)
        adjust_learning_rate_D(optimizer_D, epoch)
        # -------------------------------------------
        model_S.train()
        model_D.train()

        for i, data in enumerate(trainloader):
            images, targets = data
            if (images.size(0)==2):
                continue
            # Set mode cuda if it is enable, otherwise mode CPU
            images = images.to(device)
            targets = targets.to(device)

            # ###########################
            # #Augmentation data
            # ##########################
            is_aug=True
            if (random.random() > 0.5) and is_aug:
                images = flip(images, dim=4)
                targets = flip(targets, dim=3)  # dim of label =dim image- 1
                is_aug=False
            if random.random() > 0.5 and is_aug:
                images = flip(images, dim=3)
                targets = flip(targets, dim=2)  # dim of label =dim image- 1
                is_aug=False
            if random.random() > 0.5 and is_aug:
                images = images.transpose(3, 4)
                targets = targets.transpose(2, 3)
                is_aug=False
            if random.random() > 0.5 and is_aug:
                images = images.transpose(3, 4).flip(4)
                targets = targets.transpose(2, 3).flip(3)
                is_aug=False

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            optimizer_D.zero_grad()
            # train with real: x->D (D(x))
            D_gt = one_hot(targets).to(device)
            D_out = model_D(images,D_gt)
            D_out = D_out.squeeze()
            # ------Add smooth label----
            #D_targets = torch.empty(targets.size()).uniform_(0.8, 1.2).to(device)
            D_targets = Variable(torch.FloatTensor(targets.size()).fill_(gt_label-label_smooth)).to(device)
            # --------------------------
            loss_D_real = criterion_D(D_out, D_targets)
            loss_D_real.backward()

            # train with fake: S(z)->D (D(S(z))
            outputs, _, _, _, _ = model_S(images)
            outputs = outputs.detach() # Do not update S network
            D_out = model_D(images,F.softmax(outputs, dim=1))
            D_out = D_out.squeeze()
            # -----Add smooth predict----
            #D_targets = torch.empty(targets.size()).uniform_(0.0, 0.3).to(device)
            D_targets = Variable(torch.FloatTensor(targets.size()).fill_(pred_label-pred_smooth)).to(device)
            # ---------------------------
            loss_D_fake = criterion_D(D_out, D_targets)
            loss_D_fake.backward()
            loss_D = loss_D_real + loss_D_fake
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(S(x)))
            ############################
            optimizer_S.zero_grad()
            # z-->S(z)
            outputs, outputs1, outputs2, outputs3, outputs4 = model_S(images)
            # ---Crossentropy loss for Seg----------
            loss_seg0 = criterion_S(outputs, targets)
            # loss_seg1 = criterion_S(outputs1, targets)
            # loss_seg2 = criterion_S(outputs2, targets)
            # loss_seg3 = criterion_S(outputs3, targets)
            # loss_seg4 = criterion_S(outputs4, targets)
            # # ---------------------------
            # loss_seg = loss_seg0 + 0.1 * (loss_seg1 + loss_seg2 + loss_seg3 + loss_seg4)
            # S(z)-->D
            D_out = model_D(images,F.softmax(outputs, dim=1))  # dim=1 is class BxCxDxHxW
            D_out = D_out.squeeze()  # Returns a tensor with all the dimensions of input of size 1 removed.
            # ------Add smooth label----
            #D_targets = torch.empty(targets.size()).uniform_(0.8, 1.2).to(device)
            # --------------------------
            D_targets = Variable(torch.FloatTensor(targets.size()).fill_(gt_label)).to(device)

            loss_adv_pred = criterion_D(D_out, D_targets) # we want to food the D
            loss = loss_seg + 0.1 * loss_adv_pred
            loss.backward()
            optimizer_S.step()

            #update_vis_plot(epoch, loss_seg, epoch_plot, 'append')
            update_vis_plot(epoch+1, loss_seg, epoch_seg_plot, 'append')
            update_vis_plot(epoch+1, loss_adv_pred, epoch_adv_plot, 'append')
            update_vis_plot(epoch+1, loss_D, epoch_D_plot, 'append')


        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
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

                #combined_image=np.concatenate((predicted_val[:,24,:,:].squeeze()*20, targets_val[:, 24, :, :].squeeze() * 20),axis=1)
                #image_show('prediction--|--target', combined_image, 5)
                #cv2.waitKey(1)

                dsc = []
                for i in range(1, num_classes):  # ignore Background 0
                    dsc_i = dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)

        # Save checkpoint
        if epoch % 1000 == (1000 - 1) or epoch == (num_epoch - 1):
            torch.save(model_S.state_dict(),
                       './checkpoints/' + '%s_%s.pth' % (str(epoch + 1).zfill(5), checkpoint_name))

        is_best = dsc > best_dsc
        best_dsc = max(dsc, best_dsc)
        if (is_best):
            best_epoch=epoch+1

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3D_DenseSeg_GAN',
            'state_dict': model_S.state_dict(),
            'best_prec1': best_dsc,
            'optimizer': optimizer_S.state_dict(),
        }, is_best, filename= './checkpoints/model_best_adv.pth.tar')

        update_vis_plot(epoch + 1, torch.from_numpy(np.array(dsc)).float(), epoch_DSC_plot, 'append')


        # #-------------------Debug-------------------------
        for param_group in optimizer_S.param_groups:
            if (epoch > 10):
                print('%0.6f | %6d | %0.5f | %0.5f | %0.5f | %0.5f   | %0.5f  | %0.5f | %6d   ' % (\
                        param_group['lr'], epoch,
                        # loss_seg,
                        loss_seg.item(),
                        # loss_adv_pred,
                        loss_adv_pred.item(),
                        # loss_D,
                        loss_D.item(),
                        # loss total,
                        loss.item(),
                        #dsc for center path
                        dsc,
                        best_dsc,
                        best_epoch), end="\r")
                #time.sleep(1)
            else:
                print('%0.6f | %6d | %0.5f | %0.5f | %0.5f | %0.5f   | %0.5f  | %0.5f | %6d   ' % ( \
                    param_group['lr'], epoch,
                    # loss_seg,
                    loss_seg.item(),
                    # loss_adv_pred,
                    loss_adv_pred.item(),
                    # loss_D,
                    loss_D.item(),
                    # loss total,
                    loss.item(),
                    # dsc for center path
                    dsc,
                    best_dsc,
                    best_epoch))

    print ("\nFinished Training with best dsc: ", best_dsc , "at epoch: ", best_epoch)