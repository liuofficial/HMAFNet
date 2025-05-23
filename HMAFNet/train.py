import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from CeDiceLoss import CeDiceLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.HMAFNet import HMAFNet
import os
import scipy.io as sio
from thop import profile
import time

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener



net = HMAFNet(num_classes=N_CLASSES).cuda()


params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
print("training : ", str(len(train_ids)) + ", testing : ", str(len(test_ids)) + ", Stride_Size : ", str(Stride_Size), ", BATCH_SIZE : ", str(BATCH_SIZE))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)



criterion = CeDiceLoss(num_classes=N_CLASSES, loss_weight=[1, 1])
lr = 1e-3 # 1e-3
weight_decay = 1e-2 # 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)



def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for img, gt, gt_e in zip(test_images, test_labels, eroded_labels):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            for i, coords in enumerate(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):

    iter_ = 0
    miou_best = 0.80
    for e in range(1, epochs + 1):
        
        if scheduler is not None:
            scheduler.step()
        net.train()
        epoch_start = time.time()


        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            
            output = net(data)
            loss = criterion(output, target[:].long()) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLr: {:.6f}\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

            if iter_ % 500 == 0:
                # We validate with the largest possible stride for faster computing
                net.eval()
                miou = test(net, test_ids, all=False, stride=Stride_Size)
                net.train()
    
                if miou > miou_best:
                    best_epoch = e
                    if DATASET == 'Vaihingen':
                        torch.save(net.state_dict(), 
                                   'results/{}/vai/{}_vai_{}_e50.pth'.format(MODEL, MODEL, WINDOW_SIZE[0]))
                    elif DATASET == 'Potsdam':
                        torch.save(net.state_dict(), 
                                   'results/{}/pots/{}_pots_{}_e50.pth'.format(MODEL, MODEL, WINDOW_SIZE[0]))
                    miou_best = miou
                print('current {} epoch miou: {:.4f} | best_miou: {:.4f}'.format(e, miou, miou_best))
    print('Best performance at Epoch: {} | best_miou: {:.4f}'.format(best_epoch, miou_best))


# ------- train -------  #
time_start = time.time()
train(net, optimizer, 50, scheduler)
time_end = time.time()
print('Total Time Cost: ', time_end-time_start)

# ------- test -------  #
net.load_state_dict(torch.load(''))
net.eval()
MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32) 
print("MIoU: ", MIoU)
