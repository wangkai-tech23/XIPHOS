import glob
import time
import csv
import os
import pandas as pd
from tqdm import tqdm
def mul_to_whole(path = '../../Dataset400_5/train/nodes/0/', df_list = [],tag = 'train'):

    # if 0: #path[-2] == '0':
    #     for j in range(1,6):
    #         start = 0
    #         for i in range(num_png):
    #             df = pd.read_csv("{}{}_{}.csv".format(path, str(j), str(start + i)), encoding='utf-8', header=None)
    #             df_list.append(df)
    # else:
    df_list = []
    if 1:
        files = os.listdir(path)  # Read the folder
        num_png = len(files)
        # if tag == 'test' and path[-2] == '5':
        #     start = 3
        # elif tag =='val' and path[-2] == '5':
        #     start = 2
        # else:
        #     start = 0

        for i in range(1,1+num_png):
            df = pd.read_csv("{}{}.csv".format(path, str(i)),encoding='utf-8', header=None)
            df_list.append(df)

    df2 = pd.concat(df_list)
    print("Loading Over",path)
    return df2

'''merge'''

time_start = time.time()
Slice = ['train','test']
div = 0
for div in [0,1]:
    for tag in [1,2,3,4,5]:
        data = mul_to_whole(path = '../ROAD_show/Dataset400_5/' + Slice[div] + '/nodes/' + str(tag) + '/') #
        data.to_csv(path_or_buf='../ROAD_show/Dataset400_5/' +  Slice[div] + '/nodes/vector_'  + str(tag) + '.csv' ,index=False)
        print('0len(data)',len(data)/400)
        print('--------------------')

    print('-='*20)
    for tag in [1,2,3,4,5]:
        data = mul_to_whole(path = '../ROAD_show/Dataset400_5/' + Slice[div] + '/nodes/0_' + str(tag) + '/') #
        data.to_csv(path_or_buf='../ROAD_show/Dataset400_5/' + Slice[div] + '/nodes/vector_0_'  + str(tag) + '.csv' ,index=False)
        print('0len(data)',len(data)/400)
        print('--------------------')

time_end = time.time()
print('Using time:', time_end - time_start)