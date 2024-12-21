import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:
    def __init__(self):
        # Инициализация весов и смещений
        self.h1 = Neuron(np.array([0.5, 0.5]), 0.5)
        self.h2 = Neuron(np.array([0.5, 0.5]), 0.5)
        self.o1 = Neuron(np.array([0.5, 0.5]), 0.5)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1

# Функция ошибки (среднеквадратическая ошибка)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Данные для входа
x = np.array([2, 3])  # Входные значения

# Целевое значение
y_true = 0  # Ожидаемое значение (приближённое к 0)

# Создание экземпляра сети
network = OurNeuralNetwork()

# Подбор значений веса и смещения (гипотетически вручную, но здесь фиксировано)
network.h1.weights = np.array([-1, -1])
network.h1.bias = 1
network.h2.weights = np.array([-1, -1])
network.h2.bias = 1
network.o1.weights = np.array([1, 1])
network.o1.bias = -2

# Вычисление выхода сети
output = network.feedforward(x)

# Вычисление ошибки
loss = mse_loss(y_true, output)

print(f"Выходное значение работы нейронной сети: {output}")
print(f"Ошибка (MSE): {loss}")
