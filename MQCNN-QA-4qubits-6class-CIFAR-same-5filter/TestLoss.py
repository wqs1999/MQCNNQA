import torch
from MQCNNConvolutionLayer import MQCNN_Conv2d
import Tool



input_size = 10
TrainLoader, TestLoader, classes = Tool.DataLoader_1280(batch_size=1000, num_workers=32, input_size=input_size, category=10)
#TrainLoader, TestLoader, classes = Tool.DataLoader_1280(batch_size=1000, num_workers=32, input_size=input_size)  # 加载数据
# TrainLoader, TestLoader, classes = Tool.DataLoader_1280(batch_size=128, num_workers=32, input_size=input_size, Train_num=640, Test_num=100, category=10)  # 加
model = torch.nn.Sequential(
        MQCNN_Conv2d(in_channels=3, out_channels=5, kernel_size=2, input_size=input_size),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.BatchNorm2d(num_features=5),
        torch.nn.Flatten(start_dim=1, end_dim=3),
        # torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=5 * ((input_size - 1) // 2) ** 2, out_features=32),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=32, out_features=10),
    )

# 加载模型权重
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'Models_V1/'
criterion = torch.nn.CrossEntropyLoss().to(device)  # 定义损失函数

accuracies=[]
test_loss=[]
def test_model(i):
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    running_loss = 0.0  # 记录当前训练阶段的损失值
    path = model_path+"model_"+str(i)+".pt"
    print(path)
    weight = torch.load(path)
    # model.linear_layer.weight = weight['linear_layer.weight']
    # model.linear_layer.bias = weight['linear_layer.bias']
    # print(type(weight))
    # for e in weight:
    #     print(e)
    for i in range(len(weight)):
        model[i]=weight[i]
    model.to(device)
    with torch.no_grad():
        for data in TestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            running_loss += loss.item()
        print('10000张测试集中的准确率为: %.5f' % (correct / total))
        print('10000张测试集中损失值为: ', running_loss)
        accuracy = correct / total
        accuracies.append(accuracy.item())
        test_loss.append(running_loss)
for i in range(-1, 49):
    test_model(i+1)

# with open('./acc_and_loss/test_loss_02.txt', 'w') as f:
#     for loss in test_loss:
#         f.write(str(loss) + '\n')
# with open("acc_and_loss/acc_02.txt", "w") as f:
#     for accuracy in accuracies:
#         f.write(str(accuracy) + '\n')