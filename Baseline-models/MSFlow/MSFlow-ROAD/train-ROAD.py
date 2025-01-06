import os, random
import numpy as np
import torch
import argparse
import wandb
from memory_profiler import profile
import psutil

from train import train

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_args(c): #--mode test --class-name bottle --eval_ckpt $PATH_OF_CKPT
    parser = argparse.ArgumentParser(description='msflow')
    # parser.add_argument('--mode', default='train', type=str, help='train or test.')
    # parser.add_argument('--eval_ckpt', default='./work_dirs/CHD/', type=str, help='checkpoint path for evaluation.')
    # parser.add_argument('--class-names', default=['all'], type=str, nargs='+', help='class names for training')




    parser.add_argument('--mode', default='train', type=str, help='train or test.')
    # parser.add_argument('--mode', default='test', type=str, help='train or test.')





    parser.add_argument('--eval_ckpt', default='./work_dirs/CHD/512/best_det.pt', type=str, help='checkpoint path for evaluation.')
    parser.add_argument('--class-names', default=['bottle'], type=str, nargs='+', help='class names for training')


    parser.add_argument('--dataset', default='mvtec', type=str, choices=['mvtec', 'visa'], help='dataset name')
    parser.add_argument('--amp_enable', action='store_true', default=False, help='use amp or not.')
    parser.add_argument('--wandb_enable', action='store_true', default=False, help='use wandb for result logging or not.')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training or not.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=512, type=int, help='train batch size')
    parser.add_argument('--meta-epochs', default=3, type=int,help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=1, type=int,help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',help='number of flow blocks used in parallel flows.')
    parser.add_argument('--pro-eval', action='store_true', default=False, help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=4, type=int, help='interval for pro evaluation.')
    parser.add_argument('--device', default="cpu", type=str,help='device.') #device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(c, k, v)
    
    if c.dataset == 'mvtec':
        from datasets import MVTEC_CLASS_NAMES
        setattr(c, 'data_path', './data/MVTec')
        if c.class_names == ['all']:
            setattr(c, 'class_names', MVTEC_CLASS_NAMES)
    elif c.dataset == 'visa':
        from datasets import VISA_CLASS_NAMES
        setattr(c, 'data_path', './data/VisA_pytorch/1cls')
        if c.class_names == ['all']:
            setattr(c, 'class_names', VISA_CLASS_NAMES)
        
    c.input_size = (256, 256) if c.class_name == 'transistor' else (9, 9)
    # c.input_size = (256, 256) if c.class_name == 'transistor' else (512, 512)

    return c
@profile
def main(c):
    c = parsing_args(c)
    init_seeds(seed=c.seed)
    c.version_name = 'CHD'# 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    # c.version_name = 'msflow_wide_resnet50_2_avgpool_pl258'


    c.class_names = ['bottle']  # ,'all'
    print(c.class_names)
    for class_name in c.class_names:
        c.class_name = class_name #'bottle'
        print('-+'*5, class_name, '+-'*5)
        c.ckpt_dir = os.path.join(c.work_dir, c.version_name, str(c.batch_size) )  # './work_dirs\\CHD'
        # c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.dataset, c.class_name) #'./work_dirs\\msflow_wide_resnet50_2_avgpool_pl258\\mvtec\\bottle'
        train(c)

# CUDA_VISIBLE_DEVICES=0 python main.py --mode test --class-name bottle --eval_ckpt $PATH_OF_CKPT

if __name__ == '__main__':
    import default as c
    import time

    time_start = time.time()
    main(c)
    time_end = time.time()
    print('\nUsing time:', time_end - time_start)

    print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    info = psutil.virtual_memory()
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, "M")
    print(u'总内存：', info.total / 1024 / 1024, "M")
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())
 