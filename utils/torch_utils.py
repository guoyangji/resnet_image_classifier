"""
__file_name__   = 'torch_utils'
__author__      = 'kwok'
__time__        = '2021/6/24 15:52'
__product_name  = PyCharm
"""
import logging
import os
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(pathname)s->%(funcName)s[line:%(lineno)d] %(levelname)s | %(message)s')


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    # 如果指定设备并且设备不为 CPU
    if device and not cpu_request:
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # torch 判断是否有可用 GPU
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device {} requested'.format(device)

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        # bytes to MB
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        # 检查 batch_size 是否与设备数兼容
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using torch {} '.format(torch.__version__)
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info('Using torch {} CPU'.format(torch.__version__))
    logger.info('')
    return torch.device('cuda:0' if cuda else 'cpu')
