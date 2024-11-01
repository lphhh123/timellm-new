import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os

from models import Autoformer, DLinear, IMULLM

from data_provider.data_divide import data_divide
from data_provider.data_list import data_list
from data_provider.data_prepare import data_prepare
# from data_provider.MyDataset import MyDataset
from data_provider.MyDataset import get_list
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content


class MyDateset(Dataset):
    def __init__(self, session_list:list, label_list:list, scale=True, size=None):
        self.session_list = session_list
        self.label_list = label_list
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
        loaded_session = np.load(self.session_list[item])
        loaded_label = np.load(self.label_list[item])
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
        return len(self.session_list)

if __name__ == "__main__":

    # 设置device列表
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    parser = argparse.ArgumentParser(description='IMU-LLM')

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='IMULLM',
                        help='model name, options: [Autoformer, DLinear, IMULLM]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--root_path', type=str, default='./dataset/', help='path of the data file')
    parser.add_argument('--label_dict', type=str, default='./dataset/label.json',help='path of the label.json')
    parser.add_argument('--data_dict', type=str, default='./dataset/data.json',help='path of the data.json')
    parser.add_argument('--train_stride', type=int, required=True, default=16, help='the stride of dividing train data')
    parser.add_argument('--test_stride', type=int, required=True, default=16, help='the stride of dividing test data')
    parser.add_argument('--redivide_datasetlist', type=int, required=True, default=0, help='0 no, 1 yes')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=16, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_path', type=str, default='./gpt2', help='path of llm')  # gpt2:./gpt2


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)


    args = parser.parse_args()
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.train_stride,
            args.test_stride,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)
        
        
        dict_path = data_prepare(args)

        train_session_list, train_label_list = get_list(args, dict_path,"train")
        size = [args.seq_len, args.label_len, args.pred_len]
        train_dataset = MyDateset(session_list = train_session_list, label_list = train_label_list,size = size)
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)


        test_session_list, test_label_list = get_list(args, dict_path,"test")
        size = [args.seq_len, args.label_len, args.pred_len]
        test_dataset = MyDateset(session_list = test_session_list, label_list = test_label_list,size = size)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
       

        # 根据模型名称实例化模型
        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = IMULLM.Model(args).float()

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        # 加载prompt
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # 用于在训练深度学习模型时实现早期停止的功能。早期停止是一种防止模型过拟合的策略，具体来说，当验证损失在一段时间内没有显著改善时，训练过程将被停止。
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        # 设置学习率调度器schedule
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        # 设置损失函数
        # 均方误差损失 (MSELoss):算预测值与实际值之间的平方差，损失值越小表示模型预测的结果越准确
        # 平均绝对误差损失 (L1Loss):计算预测值与实际值之间的绝对差
        # 交叉熵损失函数（CrossEntropyLoss）:计算分类损失
        loss_MSE = nn.MSELoss()
        loss_L1 = nn.L1Loss()
        loss_CE = nn.CrossEntropyLoss()

      
        train_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
                train_loader, test_loader, model, model_optim, scheduler)
        

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            
            out_MSEloss = []
            out_L1loss = []
            out_label_CEloss = []
            in_label_CEloss = []

            model.train()
            epoch_time = time.time()

            # try:

            # except Exception as e:
            #     print("处理第{}个train_loader时发生异常".format(i))
            #     continue 

            for i, (batch_x, batch_y, batch_x_label, batch_y_label) in tqdm(enumerate(train_loader)):
                # if i % 50 == 0:
                #   print("Epoch {} iter {}".format(epoch, i))
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_label = batch_x_label.to(accelerator.device)
                batch_y_label = batch_y_label.to(accelerator.device)

                # decoder input
                # dec_inp: 创建一个与预测长度相同的全零张量，用于解码器输入。
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    accelerator.device)
                # 将真实标签的前 label_len 个时间步与全零张量拼接，以形成解码器的输入。
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs, y_hat, x_label_one_hot, y_enc, yy_lable_one_hot = model(batch_x, dec_inp, batch_x_label, batch_y_label)[0]
                        else:
                            outputs, y_hat, x_label_one_hot, y_enc, yy_lable_one_hot = model(batch_x, dec_inp, batch_x_label, batch_y_label)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                        out_seq_MSEloss = loss_MSE(outputs, batch_y)
                        out_MSEloss.append(out_seq_MSEloss.item())

                        out_seq_L1loss = loss_L1(outputs, batch_y)
                        out_L1loss.append(out_seq_L1loss.item())

                        out_label_loss = loss_CE(y_hat, x_label_one_hot)
                        out_label_CEloss.append(out_label_loss.item())

                        in_label_loss = loss_CE(y_enc, yy_lable_one_hot)
                        in_label_CEloss.append(in_label_loss.item())
                        
                        # 计算总损失：L1+CE+CE
                        loss = out_seq_L1loss + out_label_loss + in_label_loss
                else:
                    if args.output_attention:
                        outputs, y_hat, x_label_one_hot, y_enc, yy_lable_one_hot = model(batch_x, dec_inp, batch_x_label, batch_y_label)[0]
                    else:
                        outputs, y_hat, x_label_one_hot, y_enc, yy_lable_one_hot = model(batch_x, dec_inp, batch_x_label, batch_y_label)    

                    # 从 outputs 和 batch_y 中提取最后 pred_len 个时间步的数据。f_dim 的设置确保在不同特征类型下正确选择输出的维度。
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    
                    # 计算四个损失
                    # （1）输出序列的MSE损失
                    out_seq_MSEloss = loss_MSE(outputs, batch_y)
                    out_MSEloss.append(out_seq_MSEloss.item())
                    # （2）输出序列的L1损失
                    out_seq_L1loss = loss_L1(outputs, batch_y)
                    out_L1loss.append(out_seq_L1loss.item())
                    # （3）输出序列的label损失
                    out_label_loss = loss_CE(y_hat, x_label_one_hot)
                    out_label_CEloss.append(out_label_loss.item())
                    # （4） 输入序列的label损失
                    in_label_loss = loss_CE(y_enc, yy_lable_one_hot)
                    in_label_CEloss.append(in_label_loss.item())
                    
                    # 计算总损失：L1+CE+CE
                    loss = out_seq_L1loss + out_label_loss + in_label_loss
                    
                # 输出 第n*100个 iter的损失与损失和
                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i+1, epoch + 1, loss.item()))
                    accelerator.print(
                        "\tout_seq_MSEloss: {0:.7f}, out_seq_L1loss: {1:.7f}, out_label_loss: {2:.7f}, in_label_loss: {3:.7f}".format(out_seq_MSEloss, out_seq_L1loss, out_label_loss, in_label_loss))
                    avg_loss = np.average(out_L1loss) + np.average(out_label_CEloss) + np.average(in_label_CEloss)
                    accelerator.print(
                        "\tthe lasted 100 iters: avg_loss: {0:.7f}".format(avg_loss))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()
            

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            
            
            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            avg_train_loss = np.average(out_L1loss) + np.average(out_label_CEloss) + np.average(in_label_CEloss) 


          
            total_loss, avg_MSEloss, avg_L1loss, avg_outlabel_loss, avg_inlabel_loss = vali(args, accelerator, model, test_loader)
           
            
            # test_mae_loss_list = []  # vali里mae和L1一样，这里只保留了L1
            # for i in  range(len(test_loader_list)):
            #     test_data = test_loader_list[i]
            #     test_loader = test_loader_list[i]
            #     # test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion)
            #     # total_loss是混合损失的均值, avg_loss是输出序列的L1损失均值, avg_label_loss是输出序列的label损失均值, avg_enc_total_loss是输入序列的label损失均值
            #     total_loss, avg_loss, avg_label_loss, avg_enc_total_loss = vali(args, accelerator, model, test_loader, loss_MAE)
            #     test_loss_list.append(total_loss.item())
            #     # test_mae_loss_list.append(test_mae_loss.item())

            #     # 输出每个test_loader的损失情况
            #     accelerator.print(
            #     "\t {0}tets_loader| Out_L1Loss: {1:.7f} Out_labelLoss: {2:.7f} In_labelLoss: {3:.7f}".format(
            #         i, avg_loss, avg_label_loss, avg_enc_total_loss))

            # 所有test_loader的混合损失均值
            # avg_test_loss = np.average(test_loss_list)

            # test_mae_loss = np.average(test_mae_loss_list)
            # accelerator.print(
            #     "Epoch: {0} | Train Loss: {1:.7f} Test Loss: {2:.7f} MAE Loss: {3:.7f}".format(
            #         epoch + 1, train_loss, test_loss, test_mae_loss))
            # accelerator.print(
            #     "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
            #         epoch + 1, avg_loss, vali_loss, test_loss))

            accelerator.print(
                "Epoch: {0} | avg_MSEloss:{1:.7f} , avg_L1loss: {2:.7f}, avg_outlabel_loss: {3:.7f}, avg_inlabel_loss: {4:.7f}".format(
                     epoch + 1, avg_MSEloss, avg_L1loss, avg_outlabel_loss, avg_inlabel_loss))

            early_stopping(avg_test_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path
        # del_files(path)  # delete checkpoint files
        # accelerator.print('success delete checkpoints')