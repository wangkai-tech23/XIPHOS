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
time_start = time.time()
TAG = 0
filepath = '../Car Hacking Dataset/normal_16_id.csv'
path = '../Show_CHD_Split/Dataset50_40/'
Slice = ['train','test']
div = 0 # div \in [0, 1]
test_dic = [15808, 17468, 20743, 30388 , 33219]#[0,21835,25929,37985,41524]

csvreader = csv.reader(open(filepath, encoding='utf-8'))
dataset = []
line = []
i = 0
nodes = 50
batchsize = 40

for i, row in enumerate(csvreader):
    if i % nodes == 0 and i != 0:
        dataset.append(line)
        line = []
    line.append(row[1])  # Keep only the IDs to create the graph

dic_search = {'': 0}  # Create a dictionary
normalset = [];
step = 0
j = 0
normal_num = 0
count_normal = 0

for div in [1,0]:
    normal_path = path + Slice[div] +'/edges/' + str(TAG) + '/'
    if not os.path.exists(normal_path):
        os.makedirs(normal_path)

# normal_path = path + str(TAG) + '/'
# if not os.path.exists(normal_path):
#     os.makedirs(normal_path)

for row in dataset:  # ID sequence only
    i = 0
    dic_search.clear()
    for j in range(len(row)):
        if i == 0:
            i = 1
            last = row[j]
            dic_search[last] = [j]
            continue
        now = row[j]
        yz = (j, j - 1, 1)
        # = (j + count_normal * nodes, j - 1 + count_normal * nodes, 1)

        normalset.append(list(yz))
        if not (now in dic_search.keys()):
            dic_search[now] = [j]
        else:
            for k, sam in enumerate(dic_search[now]):
                if k > 4: # If the dictionary is too long, only take the last five to reduce the amount of computation
                    break
                yz = (j, sam, 1)
                # yz = (j + count_normal * nodes, sam + count_normal * nodes, 1)
                normalset.append(list(yz))
            dic_search[now].insert(0, j)
        last = now
    count_normal += 1

    div = count_normal // test_dic[TAG]
    normal_path = path + Slice[div] + '/edges/' + str(TAG) + '/'
    write_path = normal_path + str(count_normal % test_dic[TAG]) + '.csv'
    for rr in normalset:
        write_csv(write_path, 'at', rr)
    # count_normal = 0
    normalset = []
    # normal_num += 1
    # if count_normal == batchsize:


    step += 1

print('load over {} ,num_normal={}'.format(filepath, normal_num))
print('writting path',path)

time_end = time.time()
print('Using time:', time_end - time_start) #


# ''' remove normal data'''
# import shutil
# tag = 0
# path = '../../Dataset50_40/train/edges/' # Count the number of files in a folder
# orignpath = path + str(tag) +'/'
# files = os.listdir(orignpath)   # Read the folder
# num_png = len(files)       # Count the number of files in a folder
# lenval = int(num_png *0.8)
# print('location of the split data:{}, {}'.format(lenval,num_png)) # Print the number of files
#
# v_goal = path[:-12] + "val/edges/" + str(tag) + "/"
#
# if not os.path.exists(v_goal):
#     os.makedirs(v_goal)
#
# for i in range(lenval,num_png):
#     shutil.move(path + str(tag) + "/" +str(i)+".csv", v_goal)
#
#
# vfiles = os.listdir(v_goal)  # Read the folder
# num_pngv = len(vfiles)       # Print the number of files
# print('|val set|',num_pngv)