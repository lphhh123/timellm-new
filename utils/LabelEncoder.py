import pandas as pd
import json
import numpy as np
import torch


def encode_labels(label_dict, df_dataLabel):
    with open(label_dict, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    # 使用自定义的字典进行标签编码
    df_dataLabel['label'] = df_dataLabel['label'].apply(lambda x: label_dict.get(x, -1))
    return df_dataLabel


def decode_labels(label_dict, df_dataLabel):
    with open(label_dict, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    # 创建反向字典
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    if isinstance(df_dataLabel, torch.Tensor):
        df_dataLabel = df_dataLabel.cpu().numpy()

    # 使用 NumPy 向量化操作进行解码
    decoded_labels = np.vectorize(reverse_label_dict.get)(df_dataLabel)
    return decoded_labels