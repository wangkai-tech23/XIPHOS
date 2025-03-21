import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
import os

def loader_SWat(root, batch_size, window_size, stride_size,train_split,label=False):
    # data = pd.read_csv(root,sep = ';', low_memory=False)  # root = 'Data/input/SWaT_Dataset_Attack_v0.csv'  # window_size = 60
    # Timestamp = pd.to_datetime(data["Timestamp"])
    # data["Timestamp"] = Timestamp
    # data = data.set_index("Timestamp")
    # labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    # for i in list(data):
    #     data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    # data = data.drop(["Normal/Attack"] , axis = 1)
    # data = data.astype(float)
    # n_sensor = len(data.columns) # n_senor 特征长度
    # #%%
    # feature = data.iloc[:,:51]
    # scaler = StandardScaler()
    # norm_feature = scaler.fit_transform(feature)
    #
    # norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    # norm_feature = norm_feature.dropna(axis=1)
    # train_df = norm_feature.iloc[:int(train_split*len(data))]
    # train_label = labels[:int(train_split*len(data))]
    # print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))
    #
    # val_df = norm_feature.iloc[int(0.6*len(data)):int(train_split*len(data))]
    # val_label = labels[int(0.6*len(data)):int(train_split*len(data))]
    #
    # test_df = norm_feature.iloc[int(train_split*len(data)):]
    # test_label = labels[int(train_split*len(data)):]
    #
    # print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))
    #
    # if label:
    #     train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    # else:
    #     train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    #
    # val_loader = DataLoader(SWat_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(SWat_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))  # get data root path
    image_path = os.path.join(data_root, "StatGraph/BaselineModels/ROAD9_9_3-each27") # ROAD_small  # flower data set path
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),transform=transform)
    # train_dataset = train_dataset.resize(len(train_dataset[0]), -1)
    # train_dataset = torch.tensor(train_dataset).reshape(len(train_dataset[0]), -1)
    train_num = len(train_dataset)
    print(train_num)

    # # {'normal':0, 'DoS':1, 'Fuzzy':2, 'Gear':3, 'RPM':4}
    nw = 0  # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform)
    # val_num = len(val_dataset)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=transform)
    # test_num = len(test_dataset)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
    #                                               num_workers=nw)

    train_loader = DataLoader(SWat_dataset(train_dataset, [], window_size, stride_size), batch_size=batch_size,shuffle=True, num_workers=nw)
    val_loader = DataLoader(SWat_dataset(val_dataset, [], window_size, stride_size), batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(SWat_dataset(test_dataset, [], window_size, stride_size), batch_size=batch_size, shuffle=False, num_workers=nw)
    n_sensor = 243

    return train_loader, val_loader, test_loader, n_sensor

def loader_SWat_OCC(root, batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv("Data/input/SWaT_Dataset_Normal_v1.csv",sep = ',', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    #%%
    feature = data.iloc[:,:51]
    scaler = StandardScaler()

    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))

    val_df = norm_feature.iloc[int(train_split*len(data)):]
    val_label = labels[int(train_split*len(data)):]
    
    data = pd.read_csv('Data/input/SWaT_Dataset_Attack_v0.csv',sep = ';', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
 
    feature = data.iloc[:,:51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    

    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = labels[int(0.8*len(data)):]

    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))
    if label:
        train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SWat_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWat_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


class SWat_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(SWat_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data = df

        # self.idx = list(range(0,len(df),stride_size))
        self.idx = list(range(len(df)))
        self.label = list(range(len(df)))


        # self.data, self.idx, self.label = self.preprocess(df,label) #~.data:(269951,51) ~.idx:26990  ~.labels
        # self.columns = np.append(df.columns, ["Label"])  #表头 ['FIT101' 'LIT101' 'MV101' 'P101' 'P102' 'AIT201' 'AIT202' 'AIT203', 'FIT201' 'MV201' 'P201' 'P202' 'P203' 'P204' 'P205' 'P206' 'DPIT301', 'FIT301' 'LIT301' 'MV301' 'MV302' 'MV303' 'MV304' 'P301' 'P302' 'AIT401', 'AIT402' 'FIT401' 'LIT401' 'P401' 'P402' 'P403' 'P404' 'UV401' 'AIT501', 'AIT502' 'AIT503' 'AIT504' 'FIT501' 'FIT502' 'FIT503' 'FIT504' 'P501', 'P502' 'PIT501' 'PIT502' 'PIT503' 'FIT601' 'P601' 'P602' 'P603' 'Label']
        # self.timeindex = df.index[self.idx]
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
        delat_time =  df.index[end_idx]-df.index[start_idx]

        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        start_index = start_idx[idx_mask]
        
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_index ]
        return df.values, start_idx[idx_mask], np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index): #index=15402
        return self.data[index], index
        # #  N X K X L X D
        # """
        # """
        # # cc = self.data[index]
        # start = self.idx[index] #start 154020
        # end = start + self.window_size #end 154080
        # data = self.data[start]
        # data = data.reshape([self.window_size,-1, 1]) #data:60,51,1
        # return torch.FloatTensor(data).transpose(0,1), self.label[index], index

