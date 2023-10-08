import math
def activate(inputs, weights):
    sums = []
    for i in range(0, len(inputs)):
        current_sum = inputs[i] * weights[i]
        sums.append(current_sum)
    final_sum = sum(sums) # computation for h for passing through sigmoid
    return_value = 1 / (1 + math.exp(-final_sum)) # return sigmoid(h)
    return return_value
if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output =  activate(inputs, weights)
    print(output)