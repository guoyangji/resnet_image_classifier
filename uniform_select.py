"""
__author__      = 'kwok'
__time__        = '2021/6/30 15:19'
"""
import os
import cv2
import glob
import time
from tqdm import tqdm

txt_list = glob.glob(os.path.join('../dataset/images', '*.txt'))

for txt in txt_list:
    folder_name = txt.split('\\')[-1].split('.')[0]
    train_path = os.path.join('../dataset/uniform', 'train', folder_name)
    val_path = os.path.join('../dataset/uniform', 'val', folder_name)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    with open(txt, 'r+') as f:
        lines = f.readlines()
        for i in tqdm(range(len(lines)), desc='uniform => {} '.format(folder_name), ncols=100):
            data = lines[i].split(' ', 1)
            image_path = os.path.join('../dataset/images', folder_name, data[0].split('/')[-1])
            label = data[1].split(',')[:-1]
            image = cv2.imread(image_path)
            image_crop = image[int(label[1]):int(label[3]), int(label[0]):int(label[2])]
            if i < len(lines) * 0.8:
                cv2.imwrite(os.path.join(train_path, '{}.jpg'.format(int(time.time() * 1000))), image_crop)
            else:
                cv2.imwrite(os.path.join(val_path, '{}.jpg'.format(int(time.time() * 1000))), image_crop)
