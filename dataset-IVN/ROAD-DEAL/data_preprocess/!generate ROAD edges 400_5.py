''' edge(adjacent matrix) generation'''
import time
import numpy as np
import csv
import os


def write_csv(filepath, way, row):   # writing fuction
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

'''loading data'''
time_start = time.time()

''' TAG: the number of abnormal data
 Deal_correlated_signal_attack         TAG = 1
 Deal_max_speedometer_attack           TAG = 2
 Deal_reverse_light_oFF_attack         TAG = 3
 Deal_reverse_light_on_attack          TAG = 4
 Deal_max_engine_coolant_temp_attack   TAG = 5
 '''
TAG = 1
file_path_list = ['','Deal_correlated_signal_attack_','Deal_max_speedometer_attack_','Deal_reverse_light_off_attack_','Deal_reverse_light_on_attack_','Deal_max_engine_coolant_temp_attack']
Slice = ['','train','train','test']
div = 1 # div \in [1,2,3]
'''Slice:  divide data into the training set, the validation set and the test set
 train      div = 1
 val        div = 2
 test       div = 3
'''
nodes = 400
for TAG in range(1,6):
    for div in range(1,4):
        filepath = '../ROAD_raw/' + file_path_list[TAG] + str(div) + '_masquerade.csv'
        path = '../ROAD_show/Dataset400_5/' + Slice[div] + '/edges/'
        if TAG == 5:
            filepath = '../ROAD_raw/'  + file_path_list[TAG] + '_masquerade.csv'
            if div == 2:
                break

        # batchsize = 5

        csvreader = csv.reader(open(filepath, encoding='utf-8'))
        dataset = [];
        labelset = []
        line = [];
        labeline = []
        i = 0
        for i, row in enumerate(csvreader):
            if i % nodes == 0 and i != 0:
                dataset.append(line)
                labelset.append(labeline)
                line = [];
                labeline = []
            line.append(row[1])  # Keep only the IDs to create the graph
            labeline.append(int(float(row[-1])))

        label = []
        for i in range(len(labelset)):
            if 1 in labelset[i]:
                label.append(1) # Indicates that this graph contains abnormal data
            else:
                label.append(0)


        ''' label: distinguish between normal data and abnormal data'''

        attack_path = path + str(TAG) + '/'
        normal_path = path +'0_'+ str(TAG) + '/'
        if not os.path.exists(attack_path):
            os.makedirs(attack_path)
        if not os.path.exists(normal_path):
            os.makedirs(normal_path)

        dic_search = {'': 0}  # Create a dictionary
        normalset = [];
        attackset = []
        step = 0
        j = 0
        # attack_num = 0;
        # normal_num = 0
        if div == 1 or div == 3:
            count_attack = 0;
            count_normal = 0


        for row in dataset:  # ID sequence only
            i = 0
            dic_search.clear()
            if label[step] != 0:
                for j in range(len(row)):
                    if i == 0:
                        i = 1
                        last = row[j]
                        dic_search[last] = [j]
                        continue
                    now = row[j]
                    yz = (j  , j - 1 , 1)
                    attackset.append(list(yz))
                    if not (now in dic_search.keys()):
                        dic_search[now] = [j]
                    else:
                        for k, sam in enumerate(dic_search[now]):
                            if k > 4:    # If the dictionary is too long, only take the last five to reduce the amount of computation
                                break
                            yz = (j  , sam  , 1) ########################################
                            attackset.append(list(yz))
                        dic_search[now].insert(0, j)
                    last = now
                count_attack += 1
                write_path = attack_path + str(count_attack) + '.csv'
                for rr in attackset:
                    write_csv(write_path, 'at', rr)
                # count_attack = 0
                attackset = []
                # attack_num += 1


            else:
                for j in range(len(row)):
                    if i == 0:
                        i = 1
                        last = row[j]
                        dic_search[last] = [j]
                        continue
                    now = row[j]
                    yz = (j , j - 1 , 1)
                    normalset.append(list(yz))
                    if not (now in dic_search.keys()):
                        dic_search[now] = [j]
                    else:
                        for k, sam in enumerate(dic_search[now]):
                            if k > 4:    # If the dictionary is too long, only take the last five to reduce the amount of computation
                                break
                            yz = (j , sam , 1)
                            normalset.append(list(yz))
                        dic_search[now].insert(0, j)
                    last = now
                count_normal += 1

            # if count_attack == batchsize:
            #
            # if count_normal == batchsize:
                write_path = normal_path + str(count_normal) + '.csv'
                for rr in normalset:
                    write_csv(write_path, 'at', rr)
                # count_normal = 0
                normalset = []
                # normal_num += 1
            step += 1

        print('------load over {} ,num_normal={},num_attack= {}'.format(filepath, count_normal, count_attack))
        print('TAG=', TAG, 'path=',path)


print('-'*20)
# ''' splice data with tag = 5 into train, val and test'''
import os
import shutil
def rename_file(old_name, new_name):
    os.rename(old_name, new_name)

tag = 5
'''deal data in train/5/'''
path = '../ROAD_show/Dataset400_5/train/edges/' # Enter the storage folder address
orignpath = path + str(tag) +'/'

files = os.listdir(orignpath)  # Read the folder
num_png = len(files)       # Count the number of files in a folder

lentest = int(num_png *0.8)
print(lentest,num_png)   # Print the location of the split data

t_goal = path[:-12] + "test/edges/" + str(tag) + "/"

if not os.path.exists(t_goal):
    os.makedirs(t_goal)

for i in range(lentest,num_png+1):
    shutil.move(orignpath +str(i)+".csv", t_goal)

files = os.listdir(t_goal)  # Read the folder
test_num = len(files)       # Count the number of files in a folder
# rename_file(orignpath+'19.csv',orignpath+'1.csv')
for i in range(1,test_num+1):
    # 使用示例
    rename_file(t_goal+str(i+lentest-1)+'.csv',t_goal+str(i)+'.csv')

print('-'*20)


#
import os
import shutil
tag = 5
'''deal data in train/0_5/'''
path = '../ROAD_show/Dataset400_5/train/edges/0_' # Enter the storage folder address
orignpath = path + str(tag) +'/'

files = os.listdir(orignpath)  # Read the folder
num_png = len(files)       # Count the number of files in a folder

lentest = int(num_png *0.8)
print(lentest,num_png)   # Print the location of the split data
#
t_goal = path[:-14] + "test/edges/0_" + str(tag) + "/"

if not os.path.exists(t_goal):
    os.makedirs(t_goal)

for i in range(lentest,num_png+1):
    shutil.move(orignpath +str(i)+".csv", t_goal)

files = os.listdir(t_goal)  # Read the folder
test_num = len(files)       # Count the number of files in a folder
def rename_file(old_name, new_name):
    os.rename(old_name, new_name)

# lentest = 94
for i in range(1,test_num+1):
    # 使用示例
    rename_file(t_goal+str(i+lentest-1)+'.csv',t_goal+str(i)+'.csv')

time_end = time.time()
print('Using time:', time_end - time_start)

# import os
# import shutil
# def rename_file(old_name, new_name):
#     os.rename(old_name, new_name)
# tag = 4
# for tag in [1,2,3,4,5]:
#     '''deal data in train/0_5/'''
#     path = '../ROAD_show/Dataset400_5/train/edges/0_' # Enter the storage folder address
#     t_goal = path[:-14] + "train/edges/0_" + str(tag) + "/"
#     files = os.listdir(t_goal)  # Read the folder
#     test_num = len(files)       # Count the number of files in a folder
#
#     for i in range(1,test_num+1):
#         # 使用示例
#         rename_file(t_goal+str(tag) + '_'+str(i)+'.csv',t_goal+str(i)+'.csv')

