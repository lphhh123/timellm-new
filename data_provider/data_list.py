import json
import random
import os

def data_list(data_dict, redivide_datasetlist):
  datajson_path = data_dict
  # 是否需要重新打乱排序、重新生成train_list和test_list
  # 0：不需要；1：需要
  if redivide_datasetlist == 1:
      train_list, test_list = list_shuffling(datajson_path)
  else:
      with open(datajson_path, 'r') as f:
        data = json.load(f)
      train_list = data["train_list"]
      test_list = data["test_list"]
  return train_list, test_list
  

# 重新生成train_list,test_list
def list_shuffling(datajson_path):
    # 初始化新的 train_list 和 test_list
    new_train_list = []
    new_test_list = []
    with open(datajson_path, 'r') as f:
      data = json.load(f)

    # 处理 kitchen、livingroom 和 bathroom 列表
    for category in ["kitchen", "livingroom", "bathroom"]:
      train_part, test_part = split_and_add(data[category])
      new_train_list.extend(train_part)
      new_test_list.extend(test_part)
    # 处理 playVR 列表
    playVR_list = data["playVR"]
    random.shuffle(playVR_list)
    new_train_list.append(playVR_list[0])
    new_test_list.append(playVR_list[1])

    # 保存新的 train_list 和 test_list 到 和data.json同目录下的new_list.txt
    directory = os.path.dirname(datajson_path)
    new_list_path = os.path.join(directory, 'new_list.txt')
    with open(new_list_path, 'w') as f:
        json.dump({"train_list": new_train_list, "test_list": new_test_list}, f, indent=4)

    return new_train_list, new_test_list


# 定义一个函数，将一个列表打乱并分割
def split_and_add(data_list):
    random.shuffle(data_list)
    split_point = len(data_list) * 2 // 3
    return data_list[:split_point], data_list[split_point:]



# # 测试
# def data_list(datajson_path,redivide_datasetlist, seq_len, pred_len):
#   datajson_path = datajson_path
#   # 是否需要重新打乱排序、重新生成train_list和test_list
#   # 0：不需要；1：需要
#   # if args.redivide_datasetlist == 1:
#   if redivide_datasetlist == 1:
#       train_list, test_list = list_shuffling(datajson_path)
#   else:
#       with open(datajson_path, 'r') as f:
#         data = json.load(f)
#       train_list = data["train_list"]
#       test_list = data["test_list"]
#   return train_list, test_list
  
# def main():
#     datajson_path = "/home/wangtiantian/lipei/timellm/dataset/data.json"
#     train_list, test_list = data_list(datajson_path, 1, 2, 3)
#     # train_list, test_list = data_list(datajson_path, 0, 2, 3)

#     print("train_list:", train_list)
#     print("test_list:", test_list)
#     print("train_list_len:", len(train_list))
#     print("test_list_len:", len(test_list))


# if __name__ == '__main__':
#     main()