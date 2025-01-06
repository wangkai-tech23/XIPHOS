import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dataset0 import load,load_data,load_normal_data,load_extra_data
import time
from memory_profiler import profile
import time
import psutil
import os
class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft, n_h).to(device) )#.cuda())
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h).to(device)) #.cuda())

    def forward(self, feat, adj):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class Model(nn.Module):
    def __init__(self, n_in, n_h, num_layers):
        super(Model, self).__init__()
        self.mlp1 = MLP(1 * n_h, n_h)
        self.mlp2 = MLP(num_layers * n_h, n_h)
        self.gnn1 = GCN(n_in, n_h, num_layers)
        self.gnn2 = GCN(n_in, n_h, num_layers)

    def forward(self, adj, diff, feat):
        lv1, gv1 = self.gnn1(feat, adj)
        lv2, gv2 = self.gnn2(feat, diff)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2

    def embed(self, feat, adj, diff):
        __, gv1, __, gv2 = self.forward(adj, diff, feat)
        return (gv1 + gv2).detach()


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        # mm = F.softplus(- p_samples)
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_graph_loss_(l_enc1, l_enc2, batch, measure, mask):
    '''  # measure = 'JSD', mask = mask
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''


    num_graphs = l_enc2.shape[0]
    num_nodes = l_enc1.shape[1]
    num_feat = l_enc1.shape[2]

    l_enc1 = l_enc1.view(-1,num_feat);
    l_enc2 = l_enc2.view(-1, num_feat)

    pos_mask = torch.full([num_nodes, num_nodes], 1) #.to(device) #.cuda()
    neg_mask = torch.full([num_nodes, num_nodes], 1)
    for i in range(num_graphs-1):
        pos_mask = torch.block_diag(pos_mask, neg_mask)
    neg_mask = torch.ones((num_graphs*num_nodes, num_graphs*num_nodes))-pos_mask  #.to(device) #.cuda()


    res = torch.mm(l_enc1, l_enc2.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False);
    E_pos = E_pos.sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False);
    E_neg= E_neg.sum()
    E_neg = E_neg / (num_nodes * num_nodes * (num_graphs - 1) * (num_graphs - 1) )
    return E_neg - E_pos


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def global_graph_loss_(g_enc1, g_enc2, batch, measure, mask):
    '''  # measure = 'JSD', mask = mask
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc2.shape[0]
    num_feat = g_enc1.shape[1]

    pos_mask = torch.eye(num_graphs) * 1
    neg_mask = torch.ones((num_graphs, num_graphs)).to(device) - pos_mask

    res = torch.mm(g_enc1, g_enc2.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
    E_pos = E_pos.sum()
    E_pos = E_pos / num_feat
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
    E_neg= E_neg.sum()
    E_neg = E_neg / (num_feat * (num_graphs - 1))
    return E_neg - E_pos


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure, mask):
    '''  # measure = 'JSD', mask = mask
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.eye(num_nodes) * 0

    neg_mask = torch.ones((num_nodes, num_graphs)).to(device) #.cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.


    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False);
    E_pos = E_pos.sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False);
    E_neg= E_neg.sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos

hid_units = 32

@profile
def execute_test(dataset, gpu, num_layer=4, epoch=40, batch=64):
    batch_size = batch
    adj, diff, feat, labels, num_nodes = load_data(dataset,'test')


    feat = torch.FloatTensor(feat)#.to(device) # .cuda()
    diff = torch.FloatTensor(diff)#.to(device) # .cuda()
    adj = torch.FloatTensor(adj)#.to(device) # .cuda()
    labels = labels.long()#.to(device) #torch.LongTensor(labels.numpy())#.to(device) # .cuda()

    ft_size = feat[0].shape[1]

    model = Model(ft_size, hid_units, num_layer)

    gra_num = min(adj.shape[0],feat.shape[0])
    print('-'*10,'Start Execute','-'*10, 'with number of graphs = ', gra_num)
    model.load_state_dict(torch.load(f'{hid_units}IVN_upgrade_final_{batch_size}.pkl'))

    model.eval()

    features = feat[:gra_num]#.to(device) # .cuda()
    adj = adj[:gra_num]#.to(device) # .cuda()
    diff = diff[:gra_num]#.to(device) # .cuda()
    labels = labels[:gra_num]#.to(device) # .cuda()
    labels = torch.unsqueeze(torch.max(labels,dim=1).values, dim=1)

    print('- '*5,'Start embend','- '*5)
    time_start = time.time()
    embeds = model.embed(features, adj, diff)
    embeds = torch.cat([embeds, labels ],1)

    write_path = 'upgrade/batch16/SIVN_' + str(hid_units) + '_embeds_' + str(batch_size) + '_test.csv'
    np.savetxt(write_path, embeds.cpu().detach().numpy(), fmt='%.2f', delimiter=',')
    time_end = time.time()
    print('- ' * 5, 'Over embend', '- ' * 5,'with time:',time_end - time_start)

@profile
def execute_test0(dataset, gpu, num_layer=4, epoch=40, batch=64):
    batch_size = batch

    adj, diff, feat, labels, num_nodes = load_normal_data(dataset,'train')  # train里的0/文件

    feat = torch.FloatTensor(feat)  # .to(device) # .cuda()
    diff = torch.FloatTensor(diff)  # .to(device) # .cuda()
    adj = torch.FloatTensor(adj)  # .to(device) # .cuda()
    labels = labels.long()  # .to(device) #torch.LongTensor(labels.numpy())#.to(device) # .cuda()

    ft_size = feat[0].shape[1]

    model = Model(ft_size, hid_units, num_layer)

    gra_num = min(adj.shape[0], feat.shape[0])
    print('-' * 10, 'Start Execute', '-' * 10, 'with number of graphs = ', gra_num)
    model.load_state_dict(torch.load(f'{hid_units}IVN_upgrade_final_{batch_size}.pkl'))

    model.eval()

    features = feat[:gra_num]  # .to(device) # .cuda()
    adj = adj[:gra_num]  # .to(device) # .cuda()
    diff = diff[:gra_num]  # .to(device) # .cuda()
    labels = labels[:gra_num]  # .to(device) # .cuda()
    labels = torch.unsqueeze(torch.max(labels, dim=1).values, dim=1)

    print('- ' * 5, 'Start embend', '- ' * 5)
    time_start = time.time()
    embeds = model.embed(features, adj, diff)
    embeds = torch.cat([embeds, labels], 1)

    write_path = 'upgrade/batch16/IVN_' + str(hid_units) + '_embeds_' + str(batch_size) + '_test0.csv'
    np.savetxt(write_path, embeds.cpu().detach().numpy(), fmt='%.2f', delimiter=',')
    time_end = time.time()
    print('- ' * 5, 'Over embend', '- ' * 5, 'with time:', time_end - time_start)


def execute_test1234(dataset, gpu, num_layer=4, epoch=40, batch=64):
    batch_size = batch
    adj, diff, feat, labels, num_nodes = load_extra_data(dataset,'train') # train里的0_1/ ... 0_4/文件

    feat = torch.FloatTensor(feat)  # .to(device) # .cuda()
    diff = torch.FloatTensor(diff)  # .to(device) # .cuda()
    adj = torch.FloatTensor(adj)  # .to(device) # .cuda()
    labels = labels.long()  # .to(device) #torch.LongTensor(labels.numpy())#.to(device) # .cuda()

    ft_size = feat[0].shape[1]

    model = Model(ft_size, hid_units, num_layer)

    gra_num = min(adj.shape[0], feat.shape[0])
    print('-' * 10, 'Start Execute', '-' * 10, 'with number of graphs = ', gra_num)
    model.load_state_dict(torch.load(f'{hid_units}IVN_upgrade_final_{batch_size}.pkl'))

    model.eval()

    features = feat[:gra_num]  # .to(device) # .cuda()
    adj = adj[:gra_num]  # .to(device) # .cuda()
    diff = diff[:gra_num]  # .to(device) # .cuda()
    labels = labels[:gra_num]  # .to(device) # .cuda()
    labels = torch.unsqueeze(torch.max(labels, dim=1).values, dim=1)

    print('- ' * 5, 'Start embend', '- ' * 5)
    time_start = time.time()
    embeds = model.embed(features, adj, diff)
    embeds = torch.cat([embeds, labels], 1)

    write_path = 'upgrade/batch16/IVN_' + str(hid_units) + '_embeds_' + str(batch_size) + '_test'+str(dataset)+'.csv'
    np.savetxt(write_path, embeds.cpu().detach().numpy(), fmt='%.2f', delimiter=',')
    time_end = time.time()
    print('- ' * 5, 'Over embend', '- ' * 5, 'with time:', time_end - time_start)


def execute(dataset, gpu, num_layer=4, epoch=40, batch=64):
    batch_size = batch

    adj, diff, feat, labels, num_nodes = load_data(dataset, 'test')

    feat = torch.FloatTensor(feat)  # .to(device) # .cuda()
    diff = torch.FloatTensor(diff)  # .to(device) # .cuda()
    adj = torch.FloatTensor(adj)  # .to(device) # .cuda()
    labels = labels.long()  # .to(device) #torch.LongTensor(labels.numpy())#.to(device) # .cuda()

    ft_size = feat[0].shape[1]

    model = Model(ft_size, hid_units, num_layer)

    gra_num = min(adj.shape[0], feat.shape[0])
    print('-' * 10, 'Start Execute', '-' * 10, 'with number of graphs = ', gra_num)

    model.load_state_dict(torch.load(f'{hid_units}IVN_upgrade_final_{batch_size}.pkl'))
    model.eval()

    features = feat[:gra_num]  # .to(device) # .cuda()
    adj = adj[:gra_num]  # .to(device) # .cuda()
    diff = diff[:gra_num]  # .to(device) # .cuda()
    labels = labels[:gra_num]  # .to(device) # .cuda()
    labels = torch.unsqueeze(torch.max(labels, dim=1).values, dim=1)

    print('- ' * 5, 'Start embend', '- ' * 5)
    time_start = time.time()
    embeds = model.embed(features, adj, diff)
    embeds = torch.cat([embeds, labels], 1)

    write_path = 'upgrade/batch16/IVN_' + str(hid_units) + '_embeds_' + str(batch_size) + '_test.csv'
    np.savetxt(write_path, embeds.cpu().detach().numpy(), fmt='%.2f', delimiter=',')
    time_end = time.time()
    print('- ' * 5, 'Over embend', '- ' * 5, 'with time:', time_end - time_start)


device = "cpu" #("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    import time

    time_start = time.time()

    gpu = -1
    # torch.cuda.set_device(gpu)
    device = "cpu"  #
    layers = [2]
    batch = [16]
    epoch = [50]
    ds = ['0']
    seeds = [123]
    for d in ds:
        print(f'####################{d}####################')
        for l in layers:
            for b in batch:
                for e in epoch:
                    for seed in seeds:
                        torch.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        np.random.seed(seed)
                        print(f'Dataset: {d}, Layer:{l}, Batch: {b}, Epoch: {e}, Seed: {seed}')
                        modeset = ['test','test0',  'test1234']
                        for ik in [0,1]:
                            executemode = modeset[ik]
                            print('exeute-mode:',executemode)
                            if executemode== 'test':
                                execute_test(d, gpu, l, e, b)
                            elif executemode== 'test0':
                                execute_test0(d, gpu, l, e, b)
    time_end = time.time()
    print('Using time:', time_end - time_start)

    print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    info = psutil.virtual_memory()
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, "M")
    print(u'总内存：', info.total / 1024 / 1024, "M")
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())