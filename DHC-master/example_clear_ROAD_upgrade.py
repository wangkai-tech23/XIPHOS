import csv

from Kitsune import Kitsune
import numpy as np
import time
import scipy
from scipy.stats import norm
from memory_profiler import profile
import time
import psutil
import os


@profile
def main():
    Flag = "last"
    path = "../GCL-master/graph-ROAD/upgrade/"+ Flag + "/" + "ROAD_64_embeds_400test.csv" #


    packet_limit = np.Inf
    maxAE = 20 #maximum size for any autoencoder in the ensemble layer

    FMgrace = 46  #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 97  #the number of instances used to train the anomaly detector (ensemble itself)

    # Build Kitsune
    K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

    print("---Running Kitsune:")
    RMSEs = []
    i = 0
    start = time.time()
    # Here we process (train/execute) each individual packet.
    # In this way, each observation is discarded after performing process() method.

    while True:
        i+=1
        if i % 10000 == 0:
            print(i)
        rmse = K.proc_next_packet()
        if rmse == -1:
            break

        RMSEs.append(rmse)
    stop = time.time()
    print("Complete. Time elapsed: "+ str(stop - start))


    from scipy.stats import norm

    '''阈值选定'''
    alpha = 0.13
    data = np.array(RMSEs[FMgrace+ADgrace:K.FE.norml_num])
    mean, std = norm.fit(data)
    Threshold = np.abs(scipy.stats.norm.ppf(alpha) ) * std + mean

    save  = ('mean',mean, 'std',std,'alpha',alpha,'Threshold',Threshold) ; save=list(save)


    if K.FE.labels == []:
        labels = np.ones(K.FE.limit)
        for i in range(K.FE.norml_num):
            labels[i] = 0
        labels = labels[FMgrace+ADgrace:]
        print('labels',len(labels))
    else:
        labels = K.FE.labels

    predict = RMSEs[FMgrace+ADgrace:]
    features = K.AnomDetector.S_l.reshape(-1, len(K.AnomDetector.v) + 1)

    np.savetxt('results/results-ROAD.csv', features)


    predict1 = np.where(predict > Threshold, predict, 0)
    predict1 =  np.array(predict1, dtype= bool).astype(int)
    import torch
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

    length = min(len(labels),len(predict1))

    labels = torch.reshape(torch.tensor(labels[len(labels)-length:]), (-1,))
    preds = torch.reshape(torch.tensor(predict1[len(predict1)-length:]), (-1,))

    nonzero_indices = torch.nonzero(torch.tensor(labels), as_tuple=True)
    labels[nonzero_indices]=1

    acc_test = accuracy(preds, labels)
    recall_test = recall_score(labels, preds, average='macro')
    precision_test = precision_score(labels, preds, average='macro')
    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)
    roc_test = roc_auc_score(np.asarray(labels,dtype=int),preds)
    print("--------------- Threshold\n Test set results:\n",
              "accuracy = {:.4f}".format(acc_test.item()),
              "recall = {:.4f}".format(recall_test.item()),
              "precision = {:.4f}".format(precision_test.item()),
              "f1 = {:.4f}".format(f1_test.item()),
              "ROC score on ROAD dataset is {}".format(roc_test)
              )
    '''single gauss'''

    data = np.loadtxt('./results/results-ROAD.csv')
    samples = data[:-1498]
    samples0= data[-1498:]
    length = min(len(labels),len(samples0))
    labels22 = torch.reshape(torch.tensor(labels[len(labels)-length:]), (-1,))

    samples1 = np.ones(samples0.shape)
    alpha = 0.01
    Ts = np.zeros(len(samples[0]))
    for i in range(len(samples[0])):
        mean, std = norm.fit(samples[:,i])
        Threshold = np.abs(scipy.stats.norm.ppf(alpha)) * std + mean
        Ts[i] = Threshold
        samples1[:, i] =  np.where(samples0[:, i] > Threshold, samples0[:, i], 0)

    predict1 = np.array(samples1 , dtype=bool).astype(int)
    predict1 = np.max(predict1,axis=1)
    predict1 = torch.reshape(torch.tensor(predict1[len(predict1)-length:]), (-1,))

    acc_test = accuracy(predict1, labels22)
    recall_test = recall_score(labels22, predict1, average='macro')
    precision_test = precision_score(labels22, predict1, average='macro')
    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)
    roc_test = roc_auc_score(np.asarray(labels22,dtype=int),predict1)

    print("---------------single gauss\n Test set results:\n",
              "accuracy = {:.4f}".format(acc_test.item()),
              "recall = {:.4f}".format(recall_test.item()),
              "precision = {:.4f}".format(precision_test.item()),
              "f1 = {:.4f}".format(f1_test.item()),
            "ROC score on ROAD dataset is {}".format(roc_test)
              )



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

