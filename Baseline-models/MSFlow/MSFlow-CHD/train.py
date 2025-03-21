import os
import time
import datetime
import numpy as np
import wandb
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, VisADataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
import os

def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor(image) # h_list = [ (8,256,128,128), (8,512, 64,64), (8,1024,32,32) ] # 现在是
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h) # h  (8,256,128,128)----- y (8,256,64,64) # h  (8,256,5,5)----- y (8,256,3,3)
        y = torch.mean(y, dim=(2, 3), keepdim=True)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1) # (8,64,64,64)
        z, jac = parallel_flow(y, [cond, ]) # z (8,256,64,64) & jac (8) # z (8,256,3,3) & jac (8)
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list) # z_list：tuple( (8,256,3,3),(8,512,2,2)(8,1024,1,1)), fuse_jac：tensor(8)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac

def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows] # 3个模型
    fusion_flow = fusion_flow.train()  # 一个GraphlNN模型

    for sub_epoch in range(c.sub_epochs): # range(0,25)
        epoch_loss = 0.
        image_count = 0

        train_bar = tqdm(loader, file=sys.stdout)
        for idx, data in enumerate(train_bar):
            image, labels = data  # 我的x:(8,3,9,9) label:(8,)

        # for idx, (image, _, _) in enumerate(loader):

            optimizer.zero_grad()
            image = image.to(c.device) #（8,3,512,512） bath=8,so, our (8,3,9,9)
            if scaler: # None
                with autocast():
                    z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)  # z_list：tuple( (8,256,3,3),(8,512,2,2)(8,1024,1,1)),   jac：tensor(8)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                loss = loss - jac
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Train Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                epoch, sub_epoch, mean_epoch_loss, lr))
        

def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow):
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    start = time.time()
    with torch.no_grad():
        train_bar = tqdm(loader, file=sys.stdout)
        for idx, data in enumerate(train_bar):
            image, label = data  # 我的x:(8,3,9,9) label:(8,)
            mask = image.transpose(0,1) #[,0,,]#label * 0
            mask = mask[0]
            mask = mask.unsqueeze(1)
            # dd.shape

        # for idx, (image, label, mask) in enumerate(loader):
            image = image.to(c.device)

            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))  # mask: (8,1,9,9)

            z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = - 0.5 * torch.mean(z**2, 1)
                outputs_list[lvl].append(logp)
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))

            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]

        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Test Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_list, size_list


def train(c):
    
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow', 
            group=c.version_name,
            name=c.class_name)
    
    Dataset = MVTecDataset if c.dataset == 'mvtec' else VisADataset

    train_dataset = Dataset(c, is_train=True)
    test_dataset  = Dataset(c, is_train=False)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))  # get data root path
    image_path = os.path.join(data_root,"StatGraph/BaselineModels/data9_9_3-each27") # CHD_small
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform)
    train_num = len(train_dataset)
    print(train_num)
    # # {'normal':0, 'DoS':1, 'Fuzzy':2, 'Gear':3, 'RPM':4}
    nw = 0  # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=nw)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False,num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False,num_workers=nw)

    # train_loader = DataLoader(SWat_dataset(train_dataset, [], window_size, stride_size), batch_size=batch_size,
    #                           shuffle=True, num_workers=nw)
    # val_loader = DataLoader(SWat_dataset(val_dataset, [], window_size, stride_size), batch_size=batch_size,
    #                         shuffle=False, num_workers=nw)
    # test_loader = DataLoader(SWat_dataset(test_dataset, [], window_size, stride_size), batch_size=batch_size,
    #                          shuffle=False, num_workers=nw)


    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)
    # if c.wandb_enable:
    #     for idx, parallel_flow in enumerate(parallel_flows):
    #         wandb.watch(parallel_flow, log='all', log_freq=100, idx=idx)
    #     wandb.watch(fusion_flow, log='all', log_freq=100, idx=len(parallel_flows))
    params = list(fusion_flow.parameters()) #42个
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())  #加到192个
        
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)  #
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0
    if c.mode == 'test':
        print('-' * 10, 'Start Test', '-' * 10)
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)
        # gt_label_list:12384 范围0-4      outputs_list：list 1548---(8,3,3)          12384 = 1548*8
        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        # __,__,__,best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)

        gt_label_list = torch.tensor(gt_label_list)#np.array(gt_label_list);
        nonzero_indices = torch.nonzero(gt_label_list, as_tuple=True) # =
        gt_label_list[nonzero_indices]=1 #tensor([0, 0, 0,  ..., 4, 4, 4])

        print('---here1----')
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(gt_label_list, anomaly_score)
        print('fpr:{}\n,tpr:{},\n thresholds:{}\n'.format(fpr, tpr, thresholds))
        thresholds.sort()
        q1 = thresholds[int(len(thresholds) / 4)];
        q3 = thresholds[int(len(thresholds) / 4 * 3)];
        q = q3 + 1.5 * (q3 - q1)
        print('q1:{},q3:{},q{}'.format(q1, q3, q))

        from sklearn.metrics import recall_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_auc_score
        def accuracy(output, labels):
            preds = output
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

        predict = np.where(anomaly_score > q, anomaly_score, 0)  # 要求 loss_test 的值都大于q，不满足的就设置成0
        predict = np.array(predict, dtype=bool).astype(int)  # 长度54267      (, anomaly_score)

        preds = torch.tensor(predict)
        acc_test = accuracy(preds, gt_label_list)  # test_labels长度54267
        recall_test = recall_score(gt_label_list, preds, average='macro')
        precision_test = precision_score(gt_label_list, preds, average='macro')
        f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)

        print("------------- Threshold\n Test set results:\n",
              "accuracy = {:.4f}".format(acc_test.item()),
              "recall = {:.4f}".format(recall_test.item()),
              "precision = {:.4f}".format(precision_test.item()),
              "f1 = {:.4f}".format(f1_test.item()),
              )
        roc_test = roc_auc_score(np.asarray(gt_label_list, dtype=int), anomaly_score)
        # # roc_test = roc_auc_score(np.asarray(test_loader.dataset.label,dtype=int),loss_test)
        print("-----------\n The ROC score on CHD dataset is {}".format(roc_test))  # The ROC score on SWaT dataset is 0.639797355039688

        return
    
    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]  # [70,90]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma) #模型优化？
    else:
        decay_scheduler = None

    print('-'*10,'Start Training', '-'*10)

    for epoch in range(start_epoch, c.meta_epochs):  #range(0,25)
        print()  #train_loader是关键
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None) # train_meta_epoch
        # train_meta_epoch(c, epoch, val_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None) # train_meta_epoch

        # gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow) # inference_meta_epoch
        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, val_loader, extractor, parallel_flows, fusion_flow) # inference_meta_epoch

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
                eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, pro_eval)

        if c.wandb_enable:
            wandb_run.log(
                {
                    'Det.AUROC': det_auroc,
                    'Loc.AUROC': loc_auroc,
                    'Loc.PRO': loc_pro_auc
                },
                step=epoch
            )
        print('---save_weights---',save_weights)
        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)
