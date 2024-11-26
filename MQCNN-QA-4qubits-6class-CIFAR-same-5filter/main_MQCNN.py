import torch
from torch import optim
from MQCNNConvolutionLayer import MQCNN_Conv2d
import Tool
# In[]

# torch.set_default_device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    input_size = 10
    TrainLoader, TestLoader, classes = Tool.DataLoader_1280(batch_size=128, num_workers=32, input_size=input_size, Train_num=640, Test_num=100, category=10)  # 加载数据
    # Net = MQCNN_Conv2d(in_channels=3, out_channels=5, kernel_size=2, input_size=input_size)#实例化模型

    Net = torch.nn.Sequential(
        MQCNN_Conv2d(in_channels=3, out_channels=5, kernel_size=2, input_size=input_size),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.BatchNorm2d(num_features=5),
        torch.nn.Flatten(start_dim=1, end_dim=3),
        # torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=5 * ((input_size - 1) // 2) ** 2, out_features=32),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=32, out_features=10),
    )
    Net = Net.to(device=device)
    print(Net)
    # for p in Net.parameters():
    #     print(p.numel())
    # total_params = sum(p.numel() for p in Net.parameters() if p.requires_grad)
    # print(f'Total number of parameters: {total_params}') # 15825

    criterion = torch.nn.CrossEntropyLoss().to(device)  # 定义损失函数
    # optimizer = optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(Net.parameters(), lr=0.001)

    accuracies = []
    loss_all = []
    test_loss=[]
    torch.set_num_threads(64)  # 设置线程数为 8，以加速训练过程。
    for epoch in range(50):  # 指定训练的轮数为 2 轮（epoch），即遍历整个数据集两次。
        running_loss = 0.0  # 记录当前训练阶段的损失值
        for i, data in enumerate(TrainLoader, 0):
            # 输入数据
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()  # 每个 batch 开始时，将优化器的梯度缓存清零，以避免梯度累积
            # optimizer_Adam.zero_grad()

            # forward + backward
            outputs = Net(inputs)
            # print(outputs[0])


            loss = criterion(outputs, labels)  # 进行前向传播，然后计算损失函数 loss
            loss.backward()  # 自动计算损失函数相对于模型参数的梯度
            # 更新参数
            # print('[epoch:%d, step:%5d] loss: %f' % (epoch + 1, i + 1, loss.item()))
            optimizer.step()  # 使用优化器 optimizer 来更新模型的权重和偏置，以最小化损失函数
            # optimizer_Adam.step()
            # for name, param in Net.named_parameters():
            #     if param.requires_grad:  # 检查参数是否需要梯度
            #         print(f"{name}: {param.grad}")
            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            # print('[epoch:%d, step:%5d] loss: %f' % (epoch + 1, i + 1, running_loss))
            # running_loss = 0.0
            if i % 10 == 9:  # 每10个batch打印一下训练状态
                print('[epoch:%d, step:%5d] loss: %f' % (epoch + 1, i + 1, running_loss))
                loss_copy = running_loss
                running_loss = 0.0
        loss_all.append(loss_copy)

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

        print('10000张测试集中的准确率为: %.4f' % (correct / total))
        # torch.save(Net.state_dict(), "./Models_V1_02/TrainingModel{}".format(epoch))

        accuracy = correct / total
        accuracies.append(accuracy.item())

        running_loss1 = 0.0
        with torch.no_grad():
            for data in TestLoader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = Net(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                running_loss1 += loss.item()
        print('10000张测试集中损失值为: ', running_loss1)
        test_loss.append(running_loss1)
        torch.save(Net, "./Models_V1/model_{}.pt".format(epoch))

    print('Finished Training')
    with open("acc_and_loss/acc_06.txt", "w") as f:
        for accuracy in accuracies:
            f.write(str(accuracy) + '\n')
    with open("acc_and_loss/loss_06.txt", "w") as f:
        for loss in test_loss:
            f.write(str(loss) + '\n')
    # with open("./acc_and_loss/loss.txt", "w") as f:
    #     for loss_copy in loss_all:
    #         f.write(str(loss_copy) + '\n')
    # Image._show(show((data + 1) / 2).resize((200, 200)))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
