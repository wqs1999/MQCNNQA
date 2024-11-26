import pennylane as qml
import torch

# 随机型
def circuit_Ran_Rx_4qubits(inputs, weights):
    # print(inputs)
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.RX(weights[2], wires=2)
    qml.RX(weights[3], wires=3)
    qml.CRX(weights[4], wires=[1, 0])
    qml.CRX(weights[5], wires=[2, 1])
    qml.CRX(weights[6], wires=[1, 3])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_Ran_Rz_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.RZ(weights[2], wires=2)
    qml.RZ(weights[3], wires=3)
    qml.CRZ(weights[4], wires=[1, 0])
    qml.CRZ(weights[5], wires=[2, 1])
    qml.CRZ(weights[6], wires=[1, 3])
    # endregion
    return [qml.expval(qml.PauliY(i)) for i in range(4)]


def circuit_Ran_Ry_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CRY(weights[4], wires=[1, 0])
    qml.CRY(weights[5], wires=[2, 1])
    qml.CRY(weights[6], wires=[1, 3])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# 线型
def circuit_Line_Rx_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.RX(weights[2], wires=2)
    qml.RX(weights[3], wires=3)
    qml.CRX(weights[4], wires=[1, 0])
    qml.CRX(weights[5], wires=[2, 3])
    qml.CRX(weights[6], wires=[2, 1])

    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_Line_Ry_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CRY(weights[4], wires=[1, 0])
    qml.CRY(weights[5], wires=[2, 3])
    qml.CRY(weights[6], wires=[2, 1])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_Line_Rz_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.RZ(weights[2], wires=2)
    qml.RZ(weights[3], wires=3)
    qml.CRZ(weights[4], wires=[1, 0])
    qml.CRZ(weights[5], wires=[2, 3])
    qml.CRZ(weights[6], wires=[2, 1])
    # endregion
    return [qml.expval(qml.PauliY(i)) for i in range(4)]

# 环型
def circuit_Ring_Rx_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.RX(weights[2], wires=2)
    qml.RX(weights[3], wires=3)
    qml.CRX(weights[4], wires=[1, 0])
    qml.CRX(weights[5], wires=[2, 1])
    qml.CRX(weights[6], wires=[3, 2])
    qml.CRX(weights[7], wires=[0, 3])

    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_Ring_Ry_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CRY(weights[4], wires=[1, 0])
    qml.CRY(weights[5], wires=[2, 1])
    qml.CRY(weights[6], wires=[3, 2])
    qml.CRY(weights[7], wires=[0, 3])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_Ring_Rz_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.RZ(weights[2], wires=2)
    qml.RZ(weights[3], wires=3)
    qml.CRZ(weights[4], wires=[1, 0])
    qml.CRZ(weights[5], wires=[2, 1])
    qml.CRZ(weights[6], wires=[3, 2])
    qml.CRZ(weights[7], wires=[0, 3])
    # endregion
    return [qml.expval(qml.PauliY(i)) for i in range(4)]

# 异构环型
def circuit_DoubleRing_Rx_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.RX(weights[2], wires=2)
    qml.RX(weights[3], wires=3)
    qml.CRX(weights[4], wires=[3, 0])
    qml.CRX(weights[5], wires=[2, 3])
    qml.CRX(weights[6], wires=[1, 2])
    qml.CRX(weights[7], wires=[0, 1])

    qml.RX(weights[8], wires=0)
    qml.RX(weights[9], wires=1)
    qml.RX(weights[10], wires=2)
    qml.RX(weights[11], wires=3)
    qml.CRX(weights[12], wires=[3, 2])
    qml.CRX(weights[13], wires=[0, 3])
    qml.CRX(weights[14], wires=[2, 1])
    qml.CRX(weights[15], wires=[1, 0])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_DoubleRing_Ry_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CRY(weights[4], wires=[3, 0])
    qml.CRY(weights[5], wires=[2, 3])
    qml.CRY(weights[6], wires=[1, 2])
    qml.CRY(weights[7], wires=[0, 1])

    qml.RY(weights[8], wires=0)
    qml.RY(weights[9], wires=1)
    qml.RY(weights[10], wires=2)
    qml.RY(weights[11], wires=3)
    qml.CRY(weights[12], wires=[3, 2])
    qml.CRY(weights[13], wires=[0, 3])
    qml.CRY(weights[14], wires=[2, 1])
    qml.CRY(weights[15], wires=[1, 0])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_DoubleRing_Rz_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.RZ(weights[2], wires=2)
    qml.RZ(weights[3], wires=3)
    qml.CRZ(weights[4], wires=[3, 0])
    qml.CRZ(weights[5], wires=[2, 3])
    qml.CRZ(weights[6], wires=[1, 2])
    qml.CRZ(weights[7], wires=[0, 1])

    qml.RZ(weights[8], wires=0)
    qml.RZ(weights[9], wires=1)
    qml.RZ(weights[10], wires=2)
    qml.RZ(weights[11], wires=3)
    qml.CRZ(weights[12], wires=[3, 2])
    qml.CRZ(weights[13], wires=[0, 3])
    qml.CRZ(weights[14], wires=[2, 1])
    qml.CRZ(weights[15], wires=[1, 0])
    # endregion
    return [qml.expval(qml.PauliY(i)) for i in range(4)]

# 区块环型
def circuit_BlockRing_Rx_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.RX(weights[2], wires=2)
    qml.RX(weights[3], wires=3)
    qml.CRX(weights[4], wires=[2, 0])
    qml.CRX(weights[5], wires=[3, 1])

    qml.CRX(weights[6], wires=[0, 1])
    qml.CRX(weights[7], wires=[1, 0])
    qml.CRX(weights[8], wires=[2, 3])
    qml.CRX(weights[9], wires=[3, 2])

    qml.CRX(weights[10], wires=[0, 2])
    qml.CRX(weights[11], wires=[1, 3])

    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_BlockRing_Ry_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CRY(weights[4], wires=[2, 0])
    qml.CRY(weights[5], wires=[3, 1])

    qml.CRY(weights[6], wires=[0, 1])
    qml.CRY(weights[7], wires=[1, 0])
    qml.CRY(weights[8], wires=[2, 3])
    qml.CRY(weights[9], wires=[3, 2])

    qml.CRY(weights[10], wires=[0, 2])
    qml.CRY(weights[11], wires=[1, 3])
    # endregion
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def circuit_BlockRing_Rz_4qubits(inputs, weights):
    # region encoder
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    # endregion

    # region anataz
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.RZ(weights[2], wires=2)
    qml.RZ(weights[3], wires=3)
    qml.CRZ(weights[4], wires=[2, 0])
    qml.CRZ(weights[5], wires=[3, 1])

    qml.CRZ(weights[6], wires=[0, 1])
    qml.CRZ(weights[7], wires=[1, 0])
    qml.CRZ(weights[8], wires=[2, 3])
    qml.CRZ(weights[9], wires=[3, 2])

    qml.CRZ(weights[10], wires=[0, 2])
    qml.CRZ(weights[11], wires=[1, 3])
    # endregion
    return [qml.expval(qml.PauliY(i)) for i in range(4)]

if __name__ == "__main__":
    n_wires = 20
    n_layers = 10

    dev = qml.device('default.qubit', wires=n_wires)

    params_shape = (10,)
    print(type(params_shape))  # (10, 20, 3)<class 'tuple'>
    params = torch.rand(params_shape)
    print(params[0])
