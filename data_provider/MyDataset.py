from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import random
import torch
import os

from data_provider.data_list import data_list
from data_provider.data_prepare import data_prepare

class MyDateset(Dataset):
    def __init__(self, session_path:list, label_path:list, scale=True, size=None):
        self.session_path = session_path
        self.label_path = label_path
        if size == None:
            self.seq_len = 10240
            self.label_len = 512
            self.pred_len = 512
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.scale = scale
        self.x_len = self.seq_len
        self.y_len = self.label_len + self.pred_len
        self.scaler = StandardScaler()

    def __getitem__(self, item):
        loaded_session = np.load(self.session_path[item])
        loaded_label = np.load(self.label_path[item])
        # 对时序数据进行处理：acc_y +9.796 和 标准化
        loaded_session[:, 1] += 9.796
        if self.scale:
            loaded_session = loaded_session
            self.scaler.fit(loaded_session)
            loaded_session = self.scaler.transform(loaded_session) 

        seq_x = loaded_session[:self.x_len]
        seq_y = loaded_session[-self.y_len:]
        seq_x_label = loaded_label[:self.x_len]
        seq_y_label = loaded_label[-self.pred_len:]

        return seq_x, seq_y, seq_x_label, seq_y_label


    def __len__(self):
        return len(self.session_path)


def get_list(args, dict_path, flag):
    # 根据flag=test/train，选择session_{data_stride}_{seq_len}_{label_len}_{pred_len}/flag
    file_path = os.path.join(dict_path, flag)
    session_path = os.path.join(file_path,'session')
    session_list = [os.path.join(session_path, item) for item in os.listdir(session_path)]
    label_path = os.path.join(file_path,'label')
    label_list = [os.path.join(label_path, item) for item in os.listdir(label_path)]
    
    return session_list, label_list


# # 测试
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='IMU-LLM')

#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)

#     # basic config
#     parser.add_argument('--task_name', type=str, default='long_term_forecast',
#                         help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
#     parser.add_argument('--is_training', type=int, default=1, help='status')
#     parser.add_argument('--model_id', type=str, default='test', help='model id')
#     parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
#     parser.add_argument('--model', type=str, default='IMULLM',
#                         help='model name, options: [Autoformer, DLinear]')
#     parser.add_argument('--seed', type=int, default=2021, help='random seed')

#     # data loader
#     parser.add_argument('--data', type=str, default='IMU', help='dataset type')
#     parser.add_argument('--redivide_datasetlist', type=int, default='0', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='/home/lipei/project/timellm-new/dataset', help='path of the data file')
#     parser.add_argument('--label_dict', type=str, default='/home/lipei/project/timellm-new/dataset/label.json',help='path of the label_dict')
#     parser.add_argument('--data_dict', type=str, default='/home/lipei/project/timellm-new/dataset/data.json',help='path of the data.json')
#     parser.add_argument('--data_stride', type=int, default=1024, help='the stride of dividing data')
#     # parser.add_argument('--data_path', type=str, default='', help='data file')
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; '
#                              'M:multivariate predict multivariate, S: univariate predict univariate, '
#                              'MS:multivariate predict univariate')
#     parser.add_argument('--loader', type=str, default='modal', help='dataset type')
#     parser.add_argument('--freq', type=str, default='h',
#                         help='freq for time features encoding, '
#                              'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
#                              'you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=96, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#     parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
#     parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')


#     args = parser.parse_args()

#     # dataset和dataloader
#     dict_path = data_prepare(args)
#     session_list, label_list = get_list(args, dict_path,"train")
#     size = [args.seq_len, args.label_len, args.pred_len]
#     train_dataset = MyDateset(session_path = session_list, label_path =label_list,size = size)
#     train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 1)
    

#     print(len(session_list))
#     print(len(label_list))
#     print(len(train_dataset))
#     print(len(train_loader))

#     train_loader_iter = iter(train_loader)
#     # 获取第一个批次的数据
#     first_batch = next(train_loader_iter)
#     seq_x, seq_y, seq_x_label, seq_y_label = first_batch

#     print(seq_x[0].shape)
#     print(seq_y[0].shape)
#     print(seq_x_label[0].shape)
#     print(seq_y_label[0].shape)
#     print("=======================================================")
#     label = np.load('/home/lipei/project/timellm-new/dataset/session_1024_1024_96_96/train/label/label0_0.npy')
#     print(label)
#     print("label.shape:", label.shape)
#     session = np.load('/home/lipei/project/timellm-new/dataset/session_1024_1024_96_96/train/session/session0_0.npy')
#     print(session)
#     print("session.shape:", session.shape)