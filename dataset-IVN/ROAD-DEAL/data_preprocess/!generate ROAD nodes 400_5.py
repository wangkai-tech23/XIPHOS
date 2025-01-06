''' node vectors generation'''
'''one node vector = 1（ID）+ 8(data field)+  3（Number of edges, maximum degree, number of nodes）'''
import csv
import numpy as np
import os
import time

def hex_to_int(lis):  # data cleaning function
    data = []
    if len(lis) == 8:
        for row in lis:
            row = int(row, 16)
            data.append(row)
        return data
    elif len(lis) == 5:
        for j in range(3):
            lis.append('-1')
    elif len(lis) == 2:
        for j in range(6):
            lis.append('-1')
    else:
        for j in range(8 - len(lis)):
            lis.append('255')
    for row in lis:
        row = int(row, 16)
        data.append(row)
    return data


def write_csv(filepath, way, row):  # writing fuction
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


class Graph():
    def __init__(self, num_of_nodes, N= 400, directed=True):
        self.num_of_nodes = num_of_nodes  # Number of nodes
        self.directed = directed
        self.list_of_edges = []  # List of edges

        self.edge_matrix = np.zeros((N, N))  # Adjacent matrix
        self.weight_matrix = np.zeros((N, N))  # Weight matrix

        self.adjacency_list = {node: set() for node in range(num_of_nodes)}

    def add_node(self):

        self.num_of_nodes += 1

    def add_edge(self, node1, node2, weight):
        if self.edge_matrix[node1][node2]:  # If node1 and node2 are connected
            self.weight_matrix[node1][node2] += weight
            self.adjacency_list[node1] = [node1, node2, self.adjacency_list[node1][2] + weight]
        else:  # If node1 and node2 are not connected
            self.edge_matrix[node1][node2] = 1
            self.weight_matrix[node1][node2] = weight
            self.adjacency_list[node1] = [node1, node2, weight]

    def record(self):  # Number of edges, maximum degree, number of nodes
        rec = []
        rec.append(np.sum(self.edge_matrix))
        rec.append(np.max((self.weight_matrix)))
        rec.append(self.num_of_nodes)
        return rec

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
div = 1
'''Slice:  divide data into the training set, the validation set and the test set
 train      div = 1
 val        div = 2
 test       div = 3
'''
nodes = 400
for TAG in range(1,6):
    for div in range(1,4):
        filepath = '../ROAD_raw/' + file_path_list[TAG] + str(div) + '_masquerade.csv'
        path = '../ROAD_show/Dataset400_5/' + Slice[div] + '/nodes/'

        if TAG == 5:
            filepath = '../ROAD_raw/'  + file_path_list[TAG] + '_masquerade.csv'
            if div == 2:
                break
        # batch_size = 5
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
            line.append(row[1])  # # Keep only the IDs to create the graph
            labeline.append(int(float(row[-1])))

        label = []
        for i in range(len(labelset)):
            if 1 in labelset[i]:
                label.append(1)  # Indicates that this graph contains abnormal data
            else:
                label.append(0)

        actual_label = np.array(labelset)
        actual_label = actual_label.flatten()

        chakan = []
        dic_search = {'': 0}  # Create a dictionary
        node_dataset = []
        att_dataset = []
        buchong = []
        step = 0
        j = 0
        for row in dataset:  # Generate timing correlation graphs and extract the graph attributes
            i = 0
            graph = Graph(0, nodes)
            dic_search.clear()
            for now in row:
                if i == 0:
                    i = 1
                    last = now
                    continue
                if not (last in dic_search.keys()):
                    dic_search[last] = len(dic_search)
                    graph.add_node()
                if not (now in dic_search.keys()):
                    dic_search[now] = len(dic_search)
                    graph.add_node()

                graph.add_edge(dic_search[now], dic_search[last], 1)
                last = now

            buchong.append(graph.record())  # Record graph attribute
            step += 1

        attack_path = path + str(TAG) + '/'
        normal_path = path +'0_'+ str(TAG) + '/'
        if not os.path.exists(attack_path):
            os.makedirs(attack_path)
        if not os.path.exists(normal_path):
            os.makedirs(normal_path)

        normalset = [];
        attackset = []
        # attack_num = 0;
        # normal_num = 0
        if div == 1 or div == 3:
            count_attack = 0;
            count_normal = 0

        tt = []
        csvreader = csv.reader(open(filepath, encoding='utf-8'))  # Iterate again, recording the ID and payload value of the point
        for step, row in enumerate(csvreader):
            if (step + 1) > (len(buchong) * nodes):
                break
            if label[int(step / nodes)] != 0:
                tt =  hex_to_int(row[3:-1])  # Add payload
                tt.insert(0, int(row[1], 16))  # Add ID
                tt.extend(buchong[int(step  /nodes )])  # Add graph attributes
                tt.append(actual_label[step] *TAG)
                attackset.append(tt)
            else:
                tt = hex_to_int(row[3:-1])
                tt.insert(0, int(row[1], 16))
                tt.extend(buchong[int(step  / nodes )])
                tt.append(0)
                normalset.append(tt)

            if (step + 1) % nodes == 0:
                if label[int(step / nodes)] != 0:
                    count_attack += 1
                    write_path = attack_path + str(count_attack) + '.csv'
                    for rr in attackset:
                        write_csv(write_path, 'at', rr)
                    # count_attack = 0
                    attackset = []
                    # attack_num += 1

                else:
                    count_normal += 1
                    write_path = normal_path + str(count_normal) + '.csv'
                    for rr in normalset:
                        write_csv(write_path, 'at', rr)
                    # count_normal = 0
                    normalset = []
                    # normal_num += 1

            # if count_attack == batch_size:
            # if count_normal == batch_size:


        print('------load over {} ,num_normal={},num_attack= {},total num = {}'.format(filepath, count_normal,count_attack,
                                                                                 len(label)  ))
        print('TAG=',TAG,'path=',path)

print('------load over {} ,num_normal={},num_attack= {},total num = {}'.format(filepath, count_normal,count_attack,
                                                                                 len(label) ))
print('TAG=',TAG,'path=',path)



# ''' splice data with tag = 5 into train, val and test'''
import os
import shutil
tag = 5
'''deal data in train/5/'''
path = '../ROAD_show/Dataset400_5/train/nodes/' # Enter the storage folder address
orignpath = path + str(tag) +'/'

files = os.listdir(orignpath)  # Read the folder
num_png = len(files)       # Count the number of files in a folder

lentest = int(num_png *0.8)
print(lentest,num_png)   # Print the location of the split data

t_goal = path[:-12] + "test/nodes/" + str(tag) + "/"

if not os.path.exists(t_goal):
    os.makedirs(t_goal)

for i in range(lentest,num_png+1):
    shutil.move(orignpath +str(i)+".csv", t_goal)

import os
def rename_file(old_name, new_name):
    os.rename(old_name, new_name)
tag = 5
'''deal data in train/0_5/'''
path = '../ROAD_show/Dataset400_5/test/nodes/' # Enter the storage folder address
orignpath = path + str(tag) +'/'

files = os.listdir(orignpath)  # Read the folder
test_num = len(files)       # Count the number of files in a folder
# rename_file(orignpath+'19.csv',orignpath+'1.csv')
for i in range(1,test_num+1):
    # 使用示例
    rename_file(orignpath+str(i+18)+'.csv',orignpath+str(i)+'.csv')

import os
import shutil
tag = 5
'''deal data in train/0_5/'''
path = '../ROAD_show/Dataset400_5/train/nodes/0_' # Enter the storage folder address
orignpath = path + str(tag) +'/'

files = os.listdir(orignpath)  # Read the folder
num_png = len(files)       # Count the number of files in a folder

lentest = int(num_png *0.8)
print(lentest,num_png)   # Print the location of the split data
#
t_goal = path[:-14] + "test/nodes/0_" + str(tag) + "/"

if not os.path.exists(t_goal):
    os.makedirs(t_goal)

for i in range(lentest,num_png+1):
    shutil.move(orignpath +str(i)+".csv", t_goal)
def rename_file(old_name, new_name):
    os.rename(old_name, new_name)

files = os.listdir(t_goal)  # Read the folder
test_num = len(files)       # Count the number of files in a folder

for i in range(1,test_num+1):
    # 使用示例
    rename_file(t_goal+str(i+lentest-1)+'.csv',t_goal+str(i)+'.csv')
time_end = time.time()
print('Using time:', time_end - time_start)
