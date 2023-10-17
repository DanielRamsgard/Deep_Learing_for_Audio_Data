import math


def activate(inputs, weights):
    """Artificial neuron."""
    return 1 / (1 + math.exp(-sum([inputs[i] * weights[i] for i in range(len(inputs))])))


if __name__ == "__main__":
    print(activate([.5, .3, .2], [.4, .7, .2]))
