"""
__author__      = 'kwok'
__time__        = '2021/6/23 17:20'
"""
import torch
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
from utils.torch_utils import select_device
import json
import numpy as np
import os


def main(source='./images', weights='./weights/uniform_resnet50.pt', label_file='./labels/uniform.json'):
    # 查找可用 GPU
    device = select_device()
    # 图片变换初始化, 需要和验证集图片变换一样
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    json_file = open(label_file)
    class_indict = json.load(json_file)
    img_list = list()

    if not source.endswith('.jpg'):
        for img_path in os.listdir(source):
            img_list.append(os.path.join(source, img_path))
    else:
        img_list.append(source)

    # 加载模型网络结构到 GPU, num_classes 根据数据集分类情况填写
    model = models.resnet50(num_classes=len(class_indict.keys())).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    for img in img_list:
        img = Image.open(img)
        plt.imshow(img)
        # 进行图片变换
        img = data_transform(img)
        # 对数据维度进行扩充
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # 对数据的维度进行压缩
            output = torch.squeeze(model(img.to(device))).cpu()
            # 按列坐 softmax 计算
            predict = torch.softmax(output, dim=0)
            # 获取最大值索引
            predict_cla = torch.argmax(predict).numpy()

        print_res = 'class: {}, prob: {:.3} \n {}'.format(
            class_indict[str(predict_cla)],
            predict[predict_cla].numpy(),
            np.around(predict.numpy(), decimals=3)
        )
        plt.title(print_res)
        plt.show()


if __name__ == '__main__':
    main(source='./images', weights='./weights/uniform_resnet50.pt', label_file='./labels/uniform.json')
