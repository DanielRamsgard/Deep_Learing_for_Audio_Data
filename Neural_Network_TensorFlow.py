from random import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf 

# array([[0.1, 0.2], [0.2, 0.2]])
# array([[0.3], [0.4]])

def generate_dataset(num_samples, test_size):   
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)]) # inputs
    y = np.array([[i[0] + i[1]] for i in x]) # results
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size) # training set is 30 percent fo whole dataset
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.2)

    # build model 2 (one layer 2 inputs) -> 5 (one hidden layer 5 neurons) -> 1 (one output layer one neuron)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"), # all connections between layer and previous layer (this is hidden layer)
        tf.keras.layers.Dense(1, activation="sigmoid") 
    ]) # keras makes tf code easier/simpler; sequential means left to right input path

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1) # believe this is gradient descent
    model.compile(optimizer=optimizer, loss="MSE") # loss == error
    
    # train model
    model.fit(x_train, y_train, epochs=100) # training data is never seen and labeled

    # evaluate model
    print("\nModel evaluation: ") # testing data is labeled but never seen before
    model.evaluate(x_test, y_test, verbose=1) # verbose means report

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]]) # predict data is not labeled
    predictions = model.predict(data)

    print("\nSome predictions:")
    for d, p in zip(data, predictions):
        print(f"{d[0]} + {d[1]} = {p[0]}")