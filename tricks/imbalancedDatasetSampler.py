"""
__author__      = 'kwok'
__time__        = '2021/8/2 17:27'
"""
import torch
import numpy as np
import torchvision
from torch.utils.data.sampler import Sampler
import utils


# 非均衡数据集采集器
class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(dataset)
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        label_to_count = dict()
        for idx in self.indices:
            try:
                label = self._get_label(dataset, idx)
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
            except Exception as e:
                print(e)
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    @staticmethod
    def _get_label(dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        # 因为 ImageFolder 是自己定义的，所以不能使用 torchvision.datasets.ImageFolder，而是使用自定义类 utils.folder.ImageFolder
        elif dataset_type is utils.folder.ImageFolder:
            return dataset.images[idx][1]
        else:
            return np.argmax(dataset.labels[idx])

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
