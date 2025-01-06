#%%
import os
import argparse
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from memory_profiler import profile
import time
import psutil



@profile
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoint/')
    parser.add_argument('--name',default='SWaT', help='the name of dataset')

    parser.add_argument('--graph', type=str, default='None')
    parser.add_argument('--model', type=str, default='MAF')


    parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
    parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--stride_size', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



    args = parser.parse_known_args()[0]
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")


    for seed in range(15,20):
        args.seed = seed
        print(args)
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        #%%
        print("Loading dataset")
        print(args.name)
        # from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC
        from Dataset.smd_smap_msl import load_smd_smap_msl
        from Dataset.swat import loader_SWat
        from Dataset.wadi import loader_WADI
        from Dataset.psm import loader_PSM

        if args.name == 'SWaT':
            train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, \
                                                                            args.batch_size, args.window_size, args.stride_size, args.train_split)

        elif args.name == 'Wadi':
            train_loader, val_loader, test_loader, n_sensor = loader_WADI(args.data_dir, \
                                                                        args.batch_size, args.window_size, args.stride_size, args.train_split)

        elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
            train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, \
                                                                        args.batch_size, args.window_size, args.stride_size, args.train_split)

        elif args.name == 'PSM':
            train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, \
                                                                        args.batch_size, args.window_size, args.stride_size, args.train_split)



        #%%
        model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, 27, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
        # model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
        model = model.to(device)

        #%%
        from torch.nn.utils import clip_grad_value_
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_path = os.path.join(args.output_dir,args.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        loss_best = 100
        roc_max = 0

        lr = args.lr
        optimizer = torch.optim.Adam([
            {'params':model.parameters(), 'weight_decay':args.weight_decay},
            ], lr=lr, weight_decay=0.0)

        for epoch in range(40):
            print(epoch)
            loss_train = []

            model.train()
            for data,idx in train_loader:
                x,labels = data #我的x:(512,3,9,9) label:(512,) #可以是(512,27,9)
                x = x.transpose(1,3) # x (512,51,60,1) :       (60,51,1) ---- (512,51,60,1)
                x = x.reshape((x.shape[0], x.shape[1], x.shape[2]* x.shape[3],1)).to(device) # (512,9,27,1)
                # x = x.transpose(1, 2)
                # idx = idx+1

                optimizer.zero_grad()
                loss = -model(x,)

                total_loss = loss

                total_loss.backward()
                clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                loss_train.append(loss.item())



            loss_test = []
            with torch.no_grad():
                # for x,_,idx in test_loader:
                #   x = x.to(device)
                test_labels = 0; i = 0;
                for data, idx in val_loader:
                    x, labels = data
                    if i == 0:
                        test_labels = labels;
                        i = 1;
                    else:
                        test_labels  = torch.cat([test_labels,labels],0)
                    x = x.transpose(1, 3)
                    x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3], 1)).to(device)  # (512,9,27,1)


                    loss = -model.test(x, ).cpu().numpy()
                    loss_test.append(loss)
            loss_test = np.concatenate(loss_test)
            # loss_test = [-3.5652418 -3.5332453 -3.5314176 -3.5080798 -3.5089905 -3.612615, -3.463659  -3.5326216 -3.5232399 -3.5063026 -3.5459309 -3.509659, -3.453492  -3.5787055 -3.4826896 -3.4902487 -3.5308065 -3.523727, -3.5138934 -3.5154252 -3.5318599 -3.5121877 -3.4627278 -3.5695145, -3.5059247 -3.5417423 -3.5650747 -3.5225883 -3.5223658 -3.4529245, -3.4845114 -3.5735567 -3.5070782 -3.5436654 -3.5773191 -3.4968338, -3.5347397 -3.5373077 -3.4834816 -3.5681345 -3.4795828 -3.513951, -3.5362895 -3.4846296 -3.5083663 -3.4963572 -3.4735944 -3.5286834, -3.5090513 -3.533128  -3.491876  -3.4729717 -3.5523949 -3.493775, -3.5062025 -3.5437994 -3.5108163 -3.5497615 -3.4678378 -3.5338748, -3.553697  -3.5388021 -3.5021718 -3.4905639 -3.4931495 -3.5748665, -3.4573894 -3.5605125 -3.584176  -3.5237951 -3.511082  -3.4709406, -3.4919162 -3.5556574 -3.4820552 -3.564116  -3.5483098 -3.4811466, -3.5811296 -3.5310302 -3.5200362 -3.5582886 -3.5231934 -3.5364106, -3.4465559 -3.4562411 -3.5361745 -3.5075257 -3.5274255 -3.5113618,...
            nonzero_indices = torch.nonzero(test_labels, as_tuple=True) # =
            test_labels[nonzero_indices]=1 #tensor([0, 0, 0,  ..., 4, 4, 4])
            roc_test = roc_auc_score(np.asarray(test_labels,dtype=int),loss_test)
            # roc_test = roc_auc_score(np.asarray(test_loader.dataset.label,dtype=int),loss_test)


            if roc_max < roc_test:
                roc_max = roc_test
                torch.save({
                'model': model.state_dict(),
                }, f"{save_path}/ROAD-model.pth")

            roc_max = max(roc_test, roc_max)
            print(roc_max)

if __name__ == '__main__':

    time_start = time.time()
    main()
    time_end = time.time()
    print('\n\nUsing time:', time_end - time_start)

    print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    info = psutil.virtual_memory()
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, "M")
    print(u'总内存：', info.total / 1024 / 1024, "M")
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())