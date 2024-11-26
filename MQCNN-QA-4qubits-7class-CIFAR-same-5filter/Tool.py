import time

import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim

Path = '../CIFAR-10/Train/'


def DataLoad(batch_size=32, num_workers=4, input_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Resize(input_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])
    trainset = tv.datasets.CIFAR10(  # PyTorch提供的CIFAR-10数据集的类，用于加载CIFAR-10数据集。
        root=Path,  # 设置数据集存储的根目录。
        train=True,  # 指定加载的是CIFAR-10的训练集。
        download=True,  # 如果数据集尚未下载，设置为True会自动下载CIFAR-10数据集。
        transform=transform)  # 设置数据集的预处理方式。

    # 数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset,  # 指定了要加载的训练集数据，即CIFAR-10数据集。
        batch_size=batch_size,  # 每个小批量(batch)的大小是4，即每次会加载4张图片进行训练。
        shuffle=True,  # 在每个epoch训练开始前，会打乱训练集中数据的顺序，以增加训练效果。
        num_workers=num_workers,
        pin_memory=True)  # 使用2个进程来加载数据，以提高数据的加载速度。

    # 测试集
    testset = tv.datasets.CIFAR10(
        root=Path,
        train=False,
        download=True,
        transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    #classes = ('bird', 'cat', 'dog')
    classes = ('bird', 'cat', 'dog', 'deer', 'frog')

    return trainloader, testloader, classes

from collections import defaultdict
import copy
import numpy as np
np.random.seed(0)
from torch.utils.data import DataLoader, Subset
def DataLoader_1280(batch_size=32, num_workers=4, input_size=32, Train_num=640, Test_num=100, category=10):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Resize(input_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])

    # region 读取12800张训练集
    full_train_dataset = tv.datasets.CIFAR10(
        root=Path,
        train=True,
        download=True,
        transform=transform)

    # 获取所有标签
    labels = np.array(full_train_dataset.targets)
    #print(labels)
    # 为每个类别创建索引列表
    class_indices = {i: np.where(labels == i)[0] for i in range(category)}
    #print(class_indices)
    # 为每个类别随机选择1280个索引
    selected_indices = {i: np.random.choice(class_indices[i], Train_num, replace=False) for i in [0, 1, 2, 3, 4, 5, 6]}
    #print(selected_indices)
    # 将选中的索引合并为一个列表
    all_selected_indices = np.concatenate(list(selected_indices.values()))

    # 使用选中的索引创建Subset对象
    train_dataset = Subset(full_train_dataset, all_selected_indices)
    print(len(train_dataset))

    # 创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 每个小批量(batch)的大小是4，即每次会加载4张图片进行训练。
        shuffle=True,  # 在每个epoch训练开始前，会打乱训练集中数据的顺序，以增加训练效果。
        num_workers=num_workers,
        pin_memory=True)
    # 准备存储选取的索引
    # endregion 读取12800张训练集


    # region 读取1000张测试集
    full_test_dataset = tv.datasets.CIFAR10(
        root=Path,
        train=False,
        download=True,
        transform=transform)

    # 获取所有标签
    labels = np.array(full_test_dataset.targets)

    # 为每个类别创建索引列表
    class_indices = {i: np.where(labels == i)[0] for i in range(category)}

    # 为每个类别随机选择100个索引
    selected_indices = {i: np.random.choice(class_indices[i], Test_num, replace=False) for i in [0, 1, 2, 3, 4, 5, 6]}
    #print(selected_indices)
    # 将选中的索引合并为一个列表
    all_selected_indices = np.concatenate(list(selected_indices.values()))

    # 使用选中的索引创建Subset对象
    test_dataset = Subset(full_test_dataset, all_selected_indices)
    print(len(test_dataset))

    # 创建DataLoader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=category*Test_num,  # 每个小批量(batch)的大小是4，即每次会加载4张图片进行训练。
        shuffle=True,  # 在每个epoch训练开始前，会打乱训练集中数据的顺序，以增加训练效果。
        num_workers=num_workers,
        pin_memory=True)
    # 准备存储选取的索引
    # endregion 读取1000张测试集
    #classes = ('plane', 'car', 'frog', 'ship', 'truck')

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    return train_loader, test_loader, classes



def CaculateAccuracy(TestLoader, device, Net, ):
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    # 使用 torch.no_grad() 上下文管理器，表示在测试过程中不需要计算梯度，以提高速度和节约内存
    with torch.no_grad():
        for data in TestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = Net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('10000张测试集中的准确率为: %.5f' % (correct / total * 100))
if __name__=="__main__":
    DataLoader_1280(batch_size=100)