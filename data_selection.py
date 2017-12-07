import pickle as pkl
import numpy as np


def find_all_index(arr,item):
    all_index=[i for i,a in enumerate(arr) if a==item]
    return all_index


def set_label(int_label,label_len):
    label=np.zeros(label_len)
    label[int_label-1]=1.0
    label=np.array(label,dtype=int)
    return label


def select_data(data,class_num,cut_len):
    train_data_list=list()
    train_label_list=list()
    test_data_list=list()
    test_label_list=list()
    for i in range(1,class_num+1):
        all_index=find_all_index(data[1],i)
        num=0
        for j in all_index:
            if num<cut_len:
                train_data_list.append(data[0][j])
                train_label_list.append(set_label(i,class_num))
                num = num + 1
            else:
                test_data_list.append(data[0][j])
                test_label_list.append(set_label(i,class_num))
                num = num + 1
    train_data=tuple((train_data_list,train_label_list))
    test_data=tuple((test_data_list,test_label_list))

    return train_data,test_data


# img_data = pkl.load(open('facedata_pkl/Yale_face_data_15_8.pkl', 'rb'))
# train_data,test_data=select_data(img_data,15,8)
