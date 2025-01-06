''' normal edge(adjacent matrix) generation'''

import numpy as np
import csv
import os


def write_csv(filepath, way, row):
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

TAG = 1
file_path_list = ['normal_16_id','DoS_Attack_dataset','Fuzzy_Attack_dataset','Spoofing_the_drive_gear_dataset','Spoofing_the_RPM_gauge_dataset']
Slice = ['','train','val','test']
div = 1 # div \in [1,2,3]
'''Slice:  divide data into the training set and the test set
 train      div = 1
 val        div = 2
 test       div = 3
'''

for TAG in range(0,5):
    filepath = './CHD/Car Hacking Dataset/' + file_path_list[TAG] + '.csv'
    path = './CHD/CHD_Split/'
    csvreader = csv.reader(open(filepath, encoding='utf-8'))
    dataset = []
    # line = []
    for i, row in enumerate(csvreader):
        dataset.append(row)
    print(file_path_list[TAG],'-length:',i)
    data1 = dataset[:int(0.8*i)]
    write_path = path + str(TAG) + '_train.csv'
    for rr in data1:
        write_csv(write_path, 'at', rr)
    data2 = dataset[int(0.8*i):]
    write_path = path +  str(TAG) + '_test.csv'
    for rr in data2:
        write_csv(write_path, 'at', rr)
    #
