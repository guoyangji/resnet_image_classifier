"""
__author__      = 'kwok'
__time__        = '2021/6/23 17:20'
"""
import torch
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
from utils.torch_utils import select_device


def main():
    # 查找可用 GPU
    device = select_device()
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_path = "image/36.jpg"
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model = models.resnet50(num_classes=2).to(device)

    weights_path = "weights/my_resNet50.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}, prob: {:.3}".format(str(predict_cla), predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
