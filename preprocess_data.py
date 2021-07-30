"""
__file_name__   = 'preprocess_data'
__author__      = 'kwok'
__time__        = '2021/6/23 9:23'
__product_name  = PyCharm
"""

import os
import shutil
from tqdm import tqdm


def preprocess_data(data_root):
    data_file = os.listdir(os.path.join(data_root, 'dataset'))
    cat_file = list(filter(lambda x: x[:3] == 'cat', data_file))
    dog_file = list(filter(lambda x: x[:3] == 'dog', data_file))

    os.makedirs(os.path.join(data_root, 'train', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'train', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'val', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'val', 'dog'), exist_ok=True)

    for i in tqdm(range(len(cat_file))):
        pic_path = os.path.join(data_root, 'dataset', cat_file[i])
        if i < len(cat_file) * 0.9:
            obj_path = os.path.join(data_root, 'train', 'cat', cat_file[i])
        else:
            obj_path = os.path.join(data_root, 'val', 'cat', cat_file[i])
        shutil.copyfile(pic_path, obj_path)

    for j in tqdm(range(len(dog_file))):
        pic_path = os.path.join(data_root, 'dataset', dog_file[j])
        if j < len(dog_file) * 0.9:
            obj_path = os.path.join(data_root, 'train', 'dog', dog_file[j])
        else:
            obj_path = os.path.join(data_root, 'val', 'dog', dog_file[j])
        shutil.copyfile(pic_path, obj_path)


if __name__ == '__main__':
    preprocess_data('dataset/dogsvscats')
