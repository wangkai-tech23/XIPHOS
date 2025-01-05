import glob
import time
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# def mul_to_whole(path = "../Show_CHD_Split/Dataset50_40/",div = 0,TAG = 1):
#     Slice = ['train','test']
#     path = path + Slice[div] +'/nodes/' + str(TAG) +'/'
#     files = os.listdir(path)
#
#     df_list = []
#     num_png = len(files)
#     print('num_png',num_png)
#     if div == 0:
#         for i in range(1,num_png+1):  # in order
#             df = pd.read_csv("{}{}.csv".format(path,str(i)),encoding='utf-8',header=None)
#             df_list.append(df)
#     elif div == 1:
#         for i in range(num_png):  # in order
#             df = pd.read_csv("{}{}.csv".format(path,str(i)),encoding='utf-8',header=None)
#             df_list.append(df)
#     else:
#         return -1
#     df = pd.concat(df_list)
#     print("Loading Over",path)
#     return df
#
# ''' train dealing'''
# Slice = ['train','test']
# div = 0; tag =1
# for div in [0,1]:
#     for tag in range(5):
#         data = mul_to_whole(path = "../Show_CHD_Split/Dataset50_40/",div = div,TAG=tag) #
#         write_path = "../Show_CHD_Split/Dataset50_40/" + Slice[div] +'/nodes/vectors' + str(tag) +'.csv'
#         np.savetxt(write_path, data, fmt='%.7f', delimiter=',')  # 是覆盖 不是续写



#
def hebing(path = "../Show_CHD_Split/Dataset50_40/",div = 0):
    Slice = ['train', 'test']
    path = path + Slice[div] +'/nodes/' # + str(TAG) +'/'
    files = glob.glob(r'{}vectors*.csv'.format(path))  # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    print('文件个数：', num_png)
    df_list = []
    for i in range(num_png):
        df = pd.read_csv("{}vectors{}.csv".format(path,str(i)), encoding='utf-8', header=None)  # load in order
        df = df[1:]
        df_list.append(df)
    df = pd.concat(df_list)
    print("Loading Over", path)
    return df

time_start = time.time()


Slice = ['train','test']
div = 0;
for div in [0,1]:
    data = hebing(path = "../Show_CHD_Split/Dataset50_40/",div = div)  #
    write_path = "../Show_CHD_Split/Dataset50_40/" + Slice[div] +'/nodes/all_vectors.csv'
    np.savetxt(write_path, data, fmt='%.7f', delimiter=',')  # 是覆盖 不是续写

time_end = time.time()
print('Using time:', time_end - time_start)
# data = []
# data = hebing(path="../../Dataset50_40/val/nodes/", df_list=data)  #
# data.to_csv(path_or_buf="../../Dataset50_40/val_nodes.csv", index=False)
# print('val:len(data)', len(data))
# print('---------------------')
#

# def hebing_test(path="../../Dataset50_40/train/nodes/", df_list=[], tag='test'):
#     tag = path[19:-7]
#     files = glob(r'{}*.csv'.format(path))  # Read the folder
#     num_png = len(files)  # Count the number of files in a folder
#     print('文件个数：', num_png)
#     for i in range(1, num_png + 1):
#         df = pd.read_csv("{}{}_{}.csv".format(path, tag, str(i)), encoding='utf-8', header=None)   # load in order
#         df = df[1:]
#         df_list.append(df)
#     df = pd.concat(df_list)
#     print("Loading Over", path)
#     return df

# data = []
# data = hebing_test(path="../../Dataset50_40/test/nodes/", df_list=data)  #
# data.to_csv(path_or_buf="../../Dataset50_40/test_nodes.csv", index=False)
# print('test:len(data)', len(data))
# print('---------------------')
