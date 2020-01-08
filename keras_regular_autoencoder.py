from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential

def loss_euclidean(y_tilde, y):
  l = np.linalg.norm(y - y_tilde) ** 2
  return l

def main():
  (x_train, _), (x_test, _) = mnist.load_data()
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

  # this is the size of our encoded representations
  model = Sequential()
  model.add(Dense(500, input_dim=784, activation='relu'))
  model.add(Dense(300,  activation='relu'))
  model.add(Dense(32,  activation='relu'))
  model.add(Dense(300,  activation='relu'))
  model.add(Dense(500, activation='relu'))
  model.add(Dense(784))
  model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

  # training model
  model.fit(x_train, x_train, epochs=2, batch_size=1)

  predictions = model.predict(x_test)
  training_predictions = model.predict(x_train)

  train_loss = 0
  for i in range(60000):
    train_loss += loss_euclidean(training_predictions[i], x_train[i])
  print("Training loss (after training) =", train_loss)

  test_loss = 0
  for i in range(10000):
    test_loss += loss_euclidean(predictions[i], x_test[i])
  print("Testing loss =", test_loss)

  # model2 = Sequential()
  # model.add(Dense(300, input_dim=784, activation='relu'))
  # model2.add(Dense(32, activation='relu', weights=model.layers[1].get_weights()))
  # activations = model2.predict(x_test)




main()