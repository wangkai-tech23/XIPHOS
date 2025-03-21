#%%
import os
import argparse
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 
import time
from memory_profiler import profile
import time
import psutil


@profile
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='./u_s/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoint/')
    parser.add_argument('--name',default='SWaT', help='the name of dataset')

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
    parser.add_argument('--window_size', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



    args = parser.parse_known_args()[0]
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join(args.output_dir,args.name)

    from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

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



    model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, 27, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
    model = model.to(device)

    checkpoint = torch.load(f"{save_path}/ROAD-model.pth")
    model.load_state_dict(checkpoint['model'])


    model.eval()

    # print('---here0----')
    loss_test = []
    with torch.no_grad():
        test_labels = 0;
        i = 0;
        for data, idx in test_loader:
            x, labels = data  # 我的x:(512,3,9,9) label:(512,) #可以是(512,27,9)
            if i == 0:
                test_labels = labels;
                i = 1;
            else:
                test_labels = torch.cat([test_labels, labels], 0)
            x = x.transpose(1, 3)  # x (512,51,60,1) :       (60,51,1) ---- (512,51,60,1)
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3], 1)).to(device)  # (512,9,27,1)
            x = x.to(device)
            loss = -model.test(x, ).cpu().numpy()
            # loss_train = -model(x, )
            loss_test.append(loss)
        # for x, _, _ in test_loader:
        #
        #     x = x.to(device)
        #     loss = -model.test(x,).cpu().numpy()
        #     loss_test.append(loss)
    loss_test = np.concatenate(loss_test)
    nonzero_indices = torch.nonzero(test_labels, as_tuple=True) # =
    test_labels[nonzero_indices]=1

    from sklearn import metrics
    fpr,tpr,thresholds=metrics.roc_curve(test_labels,loss_test)
    print('fpr:{}\n,tpr:{},\n thresholds:{}\n'.format(fpr,tpr,thresholds) )
    thresholds.sort()
    q1 = thresholds[int(len(thresholds)/4)];


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


    predict = np.where(loss_test > q1, loss_test, 0) # 要求 loss_test 的值都大于q，不满足的就设置成0
    predict =  np.array(predict, dtype= bool).astype(int)  #长度54267

    preds = torch.tensor(predict)
    acc_test = accuracy(preds, test_labels)   #test_labels长度54267
    recall_test = recall_score(test_labels, preds, average='macro')
    precision_test = precision_score(test_labels, preds, average='macro')
    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)

    print("-------------- Threshold\n Test set results:\n",
              "accuracy = {:.4f}".format(acc_test.item()),
              "recall = {:.4f}".format(recall_test.item()),
              "precision = {:.4f}".format(precision_test.item()),
              "f1 = {:.4f}".format(f1_test.item()),
              )
    roc_test = roc_auc_score(np.asarray(test_labels, dtype=int), loss_test)
    print("-----------\n The ROC score on {} dataset is {}".format(args.name, roc_test))  


if __name__ == '__main__':
    #

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