
import os
import numpy as np
import pickle as pkl
from skimage import io,transform
import random
import math


def get_img_features(path):
    img_feature=list()
    label_list=list()
    time=1
    for file in os.listdir(path):
        if os.path.isfile(path+file):
            try:
                img=io.imread(path+file,as_grey=True)
                img=transform.resize(img,(28,28))
                img=(img-img.min())/(img.max()-img.min())
                # label=file.split('_')[0]
                # label=int(label[1:])
                label=file.split('-')[0]
                label=int(label[0:])
            except EOFError:
                print('EOFError happened at ',time,' ',file)
                return 0,0
            feature=np.array(img)
            img_feature.append(feature)
            label_list.append(label)
            time=time+1
            if(time%100==0):
                print('Samples: ',time,' ',file)
    print('Done!')
    img_data=tuple((img_feature,label_list))
    return img_data


Yale_face_data = get_img_features('facedata/IMM_face/')
pkl.dump(Yale_face_data,open('facedata_pkl/IMM_face_data_40_5.pkl','wb'))