#Check if cython code has been compiled
import os
import subprocess
import glob

use_extrapolation=False #experimental correlation code
if use_extrapolation:
    print("Importing AfterImage Cython Library")
    if not os.path.isfile("AfterImage.c"): #has not yet been compiled, so try to do so...
        cmd = "python setup.py build_ext --inplace"
        subprocess.call(cmd,shell=True)
#Import dependencies
# import netStat as ns
import csv
import numpy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess


#Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
# 使用“get_next_vector()”从给定的pcap文件中每次提取一个包的Kitsune特征
# 如果安装了wireshark (tshark)，则使用它来解析(更快)，否则使用scapy(慢得多)。
# 如果使用wireshark，则会生成一个tsv文件(pcap的解析版本)-您可以下次使用它作为输入
class FE:
    def __init__(self,file_path,limit=np.inf,num = 128):
        self.path = file_path
        self.limit = limit
        self.norml_num = 0
        self.num = num
        self.dataset = []
        self.curIndx = 0
        self.labels = []



        self.parse_type = None #unknown

        # self.length0 = 0

        self.tsvin = None #  用于解析TSV文件
        self.scapyin = None #  用于用scapy解析pcap

        ### Prep pcap ## 上一个包
        self.__prep__()

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        # self.nstat = ns.netStat(np.nan, maxHost, maxSess)
        # self.num = 0


    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return 'C:\Program Files\Wireshark\\tshark.exe'
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def __prep__(self):
        ### Find file: ###
        if not os.path.isfile(self.path):  # file does not exist
            print("File: " + self.path + " does not exist")
            raise Exception()
        ### open readers ##
        if self.parse_type == "csv":
            print("Reading csv file ...")


        mypath = self.path[:-25]  # example_clear_upgrade
        files = glob(r'{}*.csv'.format(mypath))  # Read the folder
        num_png = len(files)  # Count the number of files in a folder

        # mypath = self.path[:-28]  # example_clear
        num_png = 2
        if self.path[31:41]=="graph-ROAD":
            num_png =2
        if num_png == 2:
            csvreader = csv.reader(open(self.path[:-4] + '0.csv', encoding='utf-8'))
            # csvreader = csv.reader(open(self.path[:-4]+'_test0.csv', encoding='utf-8'))
            i = 0
            for row in csvreader:
                # if i == 0:
                #     i = 1
                #     continue
                # if i == 10000:
                #     break
                self.dataset.append(row)
                i += 1
            # self.length0 = i
            self.norml_num = i
            print('len of embeds0:', i)


            csvreader2 = csv.reader(open(self.path, encoding='utf-8'))
            i = 0
            limit = 0
            for row in csvreader2:
                self.dataset.append(row)
                limit += 1
            print('limit=', limit)

            self.dataset = np.array(self.dataset[1:]).astype(np.float32)
            # self.dataset = self.dataset[:45000]
            print('dataset.shape', self.dataset.shape)
            if 1:  # self.dataset.shape[1] == 33:
                self.labels = self.dataset[:, -1]
                self.dataset = self.dataset[:, :-1]
                for j in range(self.dataset.shape[1]):
                    self.dataset[:, j] = (self.dataset[:, j] - np.min(self.dataset[:, j])) / (
                                np.max(self.dataset[:, j]) - np.min(self.dataset[:, j]))

            # self.dataset = np.float(self.dataset)
            self.limit = len(self.dataset)

            # self.norml_num = 20000
            # self.dataset[:, 30] = 0;
            # self.dataset[:, 9] = 0;
            # self.dataset[:, 26] = 0;

            self.num = len(self.dataset[0])
            print('2dataset.shape', self.dataset.shape)
        else:
            # now_num = 0
            for step in range(num_png-2):
                csvreader = csv.reader(open(self.path[:-4] + str(step+1) +'.csv', encoding='utf-8'))
                # csvreader = csv.reader(open(self.path[:-4]+'_test0.csv', encoding='utf-8'))
                i = 0
                for row in csvreader:
                    # if i == 0:
                    #     i = 1
                    #     continue
                    # if i == 10000:
                    #     break
                    self.dataset.append(row)
                    i += 1
                # self.length0 = i
                self.norml_num += i
            print('len of all embeds0_:', self.norml_num)

            csvreader2 = csv.reader(open(self.path, encoding='utf-8'))
            i = 0
            limit = 0
            for row in csvreader2:
                self.dataset.append(row)
                limit += 1
            print('limit=', limit)

            self.dataset = np.array(self.dataset).astype(np.float32)

            print('dataset.shape', self.dataset.shape)
            if 1:  # self.dataset.shape[1] == 33:
                self.labels = self.dataset[:, -1]
                self.dataset = self.dataset[:, :-1]
                for j in range(self.dataset.shape[1]):
                    self.dataset[:, j] = (self.dataset[:, j] - np.min(self.dataset[:, j])) / (
                                np.max(self.dataset[:, j]) - np.min(self.dataset[:, j]))

            # self.dataset = np.float(self.dataset)
            self.limit = len(self.dataset)
            # self.dataset = self.dataset[~np.isnan(self.dataset)]
            # self.dataset[:,30] = 0;self.dataset[:,9] = 0;self.dataset[:,26] = 0;
            self.num = self.dataset.shape[1]
            print('dataset.shape', self.dataset.shape)


        # csvreader = csv.reader(open(self.path[:-4]+'0.csv', encoding='utf-8'))
        # # csvreader = csv.reader(open(self.path[:-4]+'_test0.csv', encoding='utf-8'))
        # i = 0
        # for row in csvreader:
        #     # if i == 0:
        #     #     i = 1
        #     #     continue
        #     # if i == 10000:
        #     #     break
        #     self.dataset.append(row)
        #     i += 1
        # # self.length0 = i
        # self.norml_num = i
        # print('len of embeds0:',i)
        #
        # csvreader2 = csv.reader(open(self.path, encoding='utf-8'))
        # i = 0
        # limit = 0
        # for row in csvreader2:
        #     self.dataset.append(row)
        #     limit += 1
        # print('limit=',limit)
        #
        # self.dataset = np.array(self.dataset).astype(np.float32)
        # print('dataset.shape',self.dataset.shape)
        # if 1: #self.dataset.shape[1] == 33:
        #     self.labels = self.dataset[:,-1]
        #     self.dataset = self.dataset[:,:-1]
        #     for j in range(self.dataset.shape[1]):
        #         self.dataset[:, j] = (self.dataset[:, j] - np.min(self.dataset[:, j])) / (np.max(self.dataset[:, j]) - np.min(self.dataset[:, j]))
        #
        #
        # # self.dataset = np.float(self.dataset)
        # self.limit = len(self.dataset)
        # self.num = len(self.dataset[0])
        # print('dataset.shape', self.dataset.shape)
    def get_next_vector(self):
        if self.curIndx == self.limit:
            return []
        x = self.dataset[self.curIndx]
        self.curIndx += 1
        return x
        ### Extract Features
        # try:
        #     x = self.nstat.updateGetStats(ID, rawfeature, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,int(framelen),float(timestamp))
        #     return x
        #
        # # MAC, IP, channel, socket
        # except Exception as e:
        #     print(e)
        #     return []


    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd =  '"' + self._tshark + '" -r '+ self.path +' -T fields '+ fields +' -E header=y -E occurrence=f > '+self.path+".tsv"
        subprocess.call(cmd,shell=True)
        print("tshark parsing complete. File saved as: "+self.path +".tsv")

    def get_num_features(self):
        return self.num  #len(self.nstat.getNetStatHeaders())
