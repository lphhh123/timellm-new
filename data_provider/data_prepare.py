import os
from data_provider.data_list import data_list
from data_provider.data_divide import data_divide


def data_prepare(args):
    seq_len = args.seq_len
    pred_len = args.pred_len
    label_len = args.label_len
    root_path = args.root_path
    train_stride = args.train_stride
    test_stride = args.test_stride
    dict_path = os.path.join(root_path, f"session_{seq_len}_{label_len}_{pred_len}_{train_stride}_{test_stride}")
    original_session_path = os.path.join(args.root_path, "all_session")
    original_label_path = os.path.join(args.root_path, "all_label")

    train_list, test_list = data_list(args.data_dict, args.redivide_datasetlist)

    # 检查文件夹是否存在,不存在则分割数据集
    if not os.path.exists(dict_path): 
        os.makedirs(dict_path)

       # 定义子文件夹名称并得到保存地址
        # train_session_save_path 例如 dataset/session_1024_512_512_96_512/train/session
        # train_label_save_path   例如 dataset/session_1024_512_512_96_512/train/label
        # test_session_save_path  例如 dataset/session_1024_512_512_96_512/test/session
        # test_label_save_path    例如 dataset/session_1024_512_512_96_512/test/label
        subfolders = ['train', 'test', 'session', 'label']
        train_session_save_path = os.path.join(dict_path, subfolders[0], subfolders[2])
        os.makedirs(train_session_save_path, exist_ok=True)
        train_label_save_path = os.path.join(dict_path, subfolders[0], subfolders[3])
        os.makedirs(train_session_save_path, exist_ok=True)
        test_session_save_path = os.path.join(dict_path, subfolders[1], subfolders[2])
        os.makedirs(test_session_save_path, exist_ok=True)
        test_label_save_path = os.path.join(dict_path, subfolders[1], subfolders[3])
        os.makedirs(test_label_save_path, exist_ok=True)
        
        for i in train_list:
            data_divide(original_session_path, original_label_path, train_session_save_path, train_label_save_path, seq_len, pred_len, label_len, train_stride, i)
        
        for j in test_list:
            data_divide(original_session_path, original_label_path, test_session_save_path, test_label_save_path, seq_len, pred_len, label_len, test_stride, j)
        print("Data is ready!")
    else:
        print(f"The folder '{dict_path}' exists.")

    return dict_path











# # 测试
# import os
# from data_list import data_list
# from data_divide import data_divide
# def data_prepare(root_path, seq_len, pred_len, label_len, data_stride):
#     # dict_path              :  dataset/session_512_1024_512
#     # original_session_path  :  dataset/all_session
#     # original_session_path  :  dataset/all_label
#     dict_path = os.path.join(root_path, f"session_{data_stride}_{seq_len}_{pred_len}")
#     original_session_path = os.path.join(root_path, "all_session")
#     original_label_path = os.path.join(root_path, "all_label")

#     # train_list, test_list = data_list(data_dict, redivide_datasetlist)
#     train_list = [10,3,5]
#     test_list = [1,4,7]

#     # 检查文件夹是否存在,不存在则分割数据集
#     if not os.path.exists(dict_path): 
#         os.makedirs(dict_path)

#         # 定义子文件夹名称并得到保存地址
#         # train_session_save_path  dataset/session_512_1024_512/train/session
#         # train_label_save_path    dataset/session_512_1024_512/train/label
#         # test_session_save_path   dataset/session_512_1024_512/test/session
#         # test_label_save_path     dataset/session_512_1024_512/test/label
#         subfolders = ['train', 'test', 'session', 'label']
#         train_session_save_path = os.path.join(dict_path, subfolders[0], subfolders[2])
#         os.makedirs(train_session_save_path, exist_ok=True)
#         train_label_save_path = os.path.join(dict_path, subfolders[0], subfolders[3])
#         os.makedirs(train_session_save_path, exist_ok=True)
#         test_session_save_path = os.path.join(dict_path, subfolders[1], subfolders[2])
#         os.makedirs(test_session_save_path, exist_ok=True)
#         test_label_save_path = os.path.join(dict_path, subfolders[1], subfolders[3])
#         os.makedirs(test_label_save_path, exist_ok=True)
        

#         for i in train_list:
#             data_divide(original_session_path, original_label_path, train_session_save_path, train_label_save_path, seq_len, pred_len, label_len, data_stride, i)
        
#         for j in test_list:
#             data_divide(original_session_path, original_label_path, test_session_save_path, test_label_save_path, seq_len, pred_len, label_len, data_stride, j)
        
        
#     else:
#         print(f"The folder '{dict_path}' exists.")

# def main():
#     root_path = '/home/lipei/project/timellm-new/dataset/'
#     seq_len = 1024
#     pred_len = 256
#     label_len = 96
#     data_stride = 1024
#     data_prepare(root_path, seq_len, pred_len, label_len, data_stride)
    
# if __name__ == '__main__':
#     main()   
    

