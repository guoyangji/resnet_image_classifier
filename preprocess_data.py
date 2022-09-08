"""
__file_name__   = 'preprocess_data'
__author__      = 'kwok'
__time__        = '2021/6/23 9:23'
__product_name  = PyCharm
"""

import os
import shutil
from tqdm import tqdm
import glob

class_dirs = ['normal', 'porn', 'sexy']


def preprocess_data(data_root):
    # data_file = os.listdir(os.path.join(data_root, 'carton_normal_porn_sexy_imgs'))

    # sexy_file = os.listdir(os.path.join(data_root, 'carton_normal_porn_sexy_imgs', 'sexy'))
    # normal_file = os.listdir(os.path.join(data_root, 'carton_normal_porn_sexy_imgs', 'normal'))
    # porn_file = os.listdir(os.path.join(data_root, 'carton_normal_porn_sexy_imgs', 'porn'))
    for class_dir in class_dirs:
        reg_file = "../dataset/carton_normal_porn_sexy_imgs/{}/*/*.jpg".format(class_dir)
        files = glob.glob(reg_file)
        os.makedirs(os.path.join(data_root, 'porn_dataset', 'train', class_dir), exist_ok=True)
        # os.makedirs(os.path.join(data_root, 'porn_dataset', 'train', 'normal'), exist_ok=True)
        # os.makedirs(os.path.join(data_root, 'porn_dataset', 'train', 'porn'), exist_ok=True)
        os.makedirs(os.path.join(data_root, 'porn_dataset', 'val', class_dir), exist_ok=True)
        # os.makedirs(os.path.join(data_root, 'porn_dataset', 'val', 'normal'), exist_ok=True)
        # os.makedirs(os.path.join(data_root, 'porn_dataset', 'val', 'porn'), exist_ok=True)

        for i in tqdm(range(len(files))):
            # pic_path = os.path.join(data_root, 'carton_normal_porn_sexy_imgs', files[i])
            name = files[i].split('/')[-1]
            if i < len(files) * 0.9:
                obj_path = os.path.join(data_root, 'porn_dataset', 'train', class_dir, name)
            else:
                obj_path = os.path.join(data_root, 'porn_dataset', 'val', class_dir, name)
            shutil.copyfile(files[i], obj_path)

        # for j in tqdm(range(len(normal_file))):
        #     pic_path = os.path.join(data_root, 'carton_normal_porn_sexy_imgs', normal_file[j])
        #     if j < len(normal_file) * 0.9:
        #         obj_path = os.path.join(data_root, 'porn_dataset', 'train', 'normal', normal_file[j])
        #     else:
        #         obj_path = os.path.join(data_root, 'porn_dataset', 'val', 'normal', normal_file[j])
        #     shutil.copyfile(pic_path, obj_path)
        #
        # for j in tqdm(range(len(porn_file))):
        #     pic_path = os.path.join(data_root, 'carton_normal_porn_sexy_imgs', porn_file[j])
        #     if j < len(porn_file) * 0.9:
        #         obj_path = os.path.join(data_root, 'porn_dataset', 'train', 'porn', porn_file[j])
        #     else:
        #         obj_path = os.path.join(data_root, 'porn_dataset', 'val', 'porn', porn_file[j])
        #     shutil.copyfile(pic_path, obj_path)


if __name__ == '__main__':
    preprocess_data('../dataset/')
