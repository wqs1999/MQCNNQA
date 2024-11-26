import torch.nn as nn
import torch
import torch.nn.functional as F

torch.cuda.set_device(1)
import Circuit

import pennylane as qml
import datetime
import matplotlib.pyplot as plt


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MQCNN_Conv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, input_size: int = 32) -> None:
        super(MQCNN_Conv2d, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = kernel_size  # 卷积核大小 kernel_size*kernel_size
        self.stride = stride  # 步长
        self.input_size = input_size
        self.dev = qml.device("default.qubit", wires=self.kernel_size ** 2)
        # self.dev = qml.device("lightning.qubit", wires=self.kernel_size ** 2)

        # region 定义卷积核参数
        self.kernels = []
        self.out = self.init_out()
        for i in range(self.out_channels):
            kernels_temp = []
            for j in range(self.in_channels):
                kernels_temp.append(self.select_filter(i))
            kernels_temp = nn.ModuleList(kernels_temp)
            self.kernels.append(kernels_temp)

        self.kernels = nn.ModuleList(self.kernels)

        # print(self.kernels)
        # endregion

        # region 定义注意力参数
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        # endregion

    # 初始化out
    def init_out(self):
        out = []
        for i in range(self.out_channels):
            out.append([])
        return out

    # 选择kernels
    def select_filter(self, i):
        if i % 5 == 0:
            weight_shapes_Ran = {"weights": (7,)}
            return qml.qnn.TorchLayer(qml.QNode(Circuit.circuit_Ran_Ry_4qubits, self.dev, diff_method="best"), weight_shapes_Ran)
        if i % 5 == 1:
            weight_shapes_Line = {"weights": (7,)}
            return qml.qnn.TorchLayer(qml.QNode(Circuit.circuit_Line_Ry_4qubits, self.dev, diff_method="best"), weight_shapes_Line)
        if i % 5 == 2:
            weight_shapes_Ring = {"weights": (8,)}
            return qml.qnn.TorchLayer(qml.QNode(Circuit.circuit_Ring_Ry_4qubits, self.dev, diff_method="best"), weight_shapes_Ring)
        if i % 5 == 3:
            weight_shapes_DoubleRing = {"weights": (16,)}
            return qml.qnn.TorchLayer(qml.QNode(Circuit.circuit_DoubleRing_Ry_4qubits, self.dev, diff_method="best"), weight_shapes_DoubleRing)
        if i % 5 == 4:
            weight_shapes_BlockRing = {"weights": (12,)}
            return qml.qnn.TorchLayer(qml.QNode(Circuit.circuit_BlockRing_Ry_4qubits, self.dev, diff_method="best"), weight_shapes_BlockRing)

    # 执行kernels
    def CarryParallelKernel(self, kernels, inputs, index):
        # torch.Size([128, 3, 4]) inputs

        out = []
        for i in range(len(inputs[0])):
            out.append(kernels[i](inputs[:, i:i + 1, ]))
        # print(out[0].shape) # torch.Size([128, 1])
        out = torch.stack(out, dim=1)
        out = torch.sum(out, dim=1)
        # print(index)
        # print("kernels:{}".format(out.shape))
        self.out[index].append(out)

    # 向前传播
    def forward(self, inputs):
        # print("inputs[0][0]:{}".format(inputs[0][0]))
        # output_cpu = inputs.to("cpu")
        # image = output_cpu.detach()[0].permute(1, 2, 0).numpy()
        # plt.imshow(image)
        # plt.show()
        # 2 2 3 3

        # region configure
        batch_size = len(inputs)
        in_channel_size = len(inputs[0])
        out_channel_size = self.out_channels
        new_image_length = len(inputs[0][0]) - self.kernel_size + 1
        new_image_width = len(inputs[0][0][0]) - self.kernel_size + 1
        # endregion

        # region 特征提取
        for q in range(new_image_length):
            for p in range(new_image_width):
                input_view = inputs[:, :, q:q + 2, p:p + 2].reshape([batch_size, in_channel_size, self.kernel_size ** 2])
                for index in range(len(self.kernels)):
                    self.CarryParallelKernel(kernels=self.kernels[index], inputs=input_view, index=index)

        output = []
        for index in range(len(self.kernels)):
            output.append(torch.stack(self.out[index], dim=1).squeeze())  # torch.Size([128, 225])
        # print(output[0].shape)
        self.out = self.init_out()
        # endregion

        output = torch.stack(output, dim=1).squeeze()
        # print(output.shape)
        output = output.reshape([batch_size, out_channel_size, new_image_length, new_image_width])
        # output_cpu = output.to("cpu")
        # print("output[0][0]:{}".format(output[0][0]))
        # image = output_cpu.detach()[0].permute(1, 2, 0).numpy()
        # for i in range(len(image[0][0])):
        #     plt.imshow(image[:, :, i:i + 1].squeeze(), cmap='gray')
        #     plt.show()
        # print(output.shape)
        return output


if __name__ == '__main__':
    # m = MQCNN_Conv2d(1, 2, 3, 4)
    # print(datetime.datetime.now())
    # print(m(torch.tensor([0.5, 0.5, 0.5, 0.5])))
    # print(m.state_dict())
    # torch.save(m.state_dict(), "./SAVE_PATH/TrainingModel")
    # print(datetime.datetime.now())
    # 2 3 2
    tensor1 = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                            [[7, 8], [9, 10], [11, 12]]], dtype=int)
    # 2 3
    tensor2 = torch.tensor([[1, 2, 3],
                            [4, 5, 6]], dtype=int)
    tensor2 = tensor2.reshape([2, 3, 1])
    print(tensor1)
    print(tensor2.expand_as(tensor1))

    # 如果你想在维度0（行）上对多个张量进行求和，可以使用torch.sum
    print(tensor1 * tensor2)
    sum_result = torch.sum(tensor1 * tensor2, dim=1)
    print(sum_result)
