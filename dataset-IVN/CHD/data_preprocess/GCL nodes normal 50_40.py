''' node vectors generation'''
'''one node vector = 1（ID）+ 8(data field)+  3（Number of edges, maximum degree, number of nodes）'''
import csv
import numpy as np
import time
import os


def To_int(lis):  # data cleaning function
    data = []
    if len(lis) == 8:
        for row in lis:
            row = int(float(row))
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
        row = int(float(row))
        data.append(row)
    return data


def write_csv(filepath, way, row): # writing fuction
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


class Graph():
    def __init__(self, num_of_nodes, N= 50, directed=True):
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
time_start = time.time()
batch_size = 40
nnodes = 50

filepath = '../Car Hacking Dataset/normal_16_id.csv'
path = '../Show_CHD_Split/Dataset50_40/'
TAG = 0
Slice = ['train','test']
div = 0 # div \in [0, 1]
test_dic = [15808, 17468, 20743, 30388 , 33219]#[0,21835,25929,37985,41524]

csvreader = csv.reader(open(filepath, encoding='utf-8'))
dataset = [];
labelset = []
line = [];
labeline = []
i = 0
for i, row in enumerate(csvreader):
    if i % nnodes == 0 and i != 0:
        dataset.append(line)
        line = []
    line.append(row[1])  # Keep only the IDs to create the graph

chakan = []
dic_search = {'': 0}  # Create a dictionary
node_dataset = []
att_dataset = []
buchong = []
step = 0
j = 0
for row in dataset:  # Generate timing correlation graphs and extract the graph attributes
    i = 0
    graph = Graph(0, nnodes)
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

    buchong.append(graph.record()) # Record graph attribute
    step += 1

normalset = [];
attackset = []
attack_num = 0;
normal_num = 0
count_attack = 0;
count_normal = 0

for div in [1,0]:
    normal_path = path + Slice[div] +'/nodes/' + str(TAG) + '/'
    if not os.path.exists(normal_path):
        os.makedirs(normal_path)

tt = []
csvreader = csv.reader(open(filepath, encoding='utf-8')) # Iterate again, recording the ID and payload value of the point
for step, row in enumerate(csvreader):

    if (step + 1) > (len(buchong) * nnodes):
        # print('step', step)
        break
    tt = row[3:]
    while '' in tt:
        tt.remove('')
    tt = To_int(tt)  # Add payload
    tt.insert(0, int(row[1], 16))  # Add ID
    tt.extend(buchong[int(step / nnodes)])  # Add graph attributes
    tt.append(0)
    normalset.append(tt)

    if (step + 1) % nnodes == 0:
        count_normal += 1

        div = count_normal // test_dic[TAG]
        normal_path = path + Slice[div] + '/nodes/' + str(TAG) + '/'
        write_path = normal_path + str(count_normal % test_dic[TAG]) + '.csv'
        for rr in normalset:
            write_csv(write_path, 'at', rr)
        # count_normal = 0
        normalset = []
        normal_num += 1

print('load over {} ,num_attack= {},num_normal={},total num = {}'.format(filepath, count_attack, count_normal,
                                                                         len(buchong) / batch_size ))


time_end = time.time()
print('Using time:', time_end - time_start)


# ''' remove normal data'''
# import os
# import shutil
# tag = 0
# path = '../../Dataset50_40/train/nodes/' # Count the number of files in a folder
# orignpath = path + str(tag) +'/'
# files = os.listdir(orignpath)   # Read the folder
# num_png = len(files)       # Count the number of files in a folder
# lenval = int(num_png *0.8)
# print('location of the split data:{}, {}'.format(lenval,num_png)) # Print the number of files
#
# v_goal = path[:-12] + "val/nodes/" + str(tag) + "/"
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

