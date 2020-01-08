import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle


def fc(x, w, b):
  # x = (m,1), label = (n,1), w = (n,m), b = (n,1)
  # y = (n,1)
  z = np.reshape(x, (-1, 1))
  y = np.matmul(w, z)
  y = np.reshape(y, (-1, 1)) + np.reshape(b, (-1, 1))
  return y


def fc_backward(dl_dy, x, w, b, y):
  # dl_dx = dl_dy * dy_dx --> dyj_dxi = wji for each j
  # dl_dxi = sum_j(dl_dyj * wji)
  n = w.shape[0]
  m = w.shape[1]
  # x = (m,1), dl_dy = (n,1), w = (n,m), dl_dx = (m,1), dl_dw = (n,m)
  dl_dx = np.zeros((n, m))
  for i, row in enumerate(w):
    dl_dx[i] = dl_dy[i][0] * row
  dl_dx = np.sum(dl_dx, axis=0)
  dl_dx = np.reshape(dl_dx, (-1, 1))
  dl_dy = np.reshape(dl_dy, (-1, 1))
  # dl_dw = dl_dy * dy_dw --> dyj_dwij = xi

  # Non-regularized:
  #dl_dw = np.matmul(dl_dy, np.reshape(x, (1, -1)))
  #dl_db = dl_dy

  # Regularized
  l2_param = 1
  dl_dw = np.matmul(dl_dy, np.reshape(x, (1, -1))) + (l2_param * w)
  # dl_db = dl_dy * dyj_dbj --> dyj_dbj = 1
  dl_db = dl_dy.reshape(-1, 1) + (l2_param * b.reshape(-1, 1))

  return dl_dx, dl_dw, dl_db


def relu(x):
  y = np.copy(x)
  y_shape = y.shape
  y = np.maximum(x, 0)
  return y


def relu_backward(dl_dy, x, y):
  # dl_dx = dl_dy * dy_dx --> if x<0 --> dy_dx=0 hence dl_dx = 0, x>=0 -> dy_dx=1 hence dl_dx = dl_dy
  # dl_dy = (z,1), dl_dx = (z,1), x = (z,1)
  x_copy = np.copy(x)
  x_copy[x_copy < 0] = 0
  x_copy[x_copy >= 0] = 1
  # dl_dx = np.reshape(dl_dy, (-1,1)) * np.reshape(x_copy, (-1,1))
  dl_dx = dl_dy * x_copy
  return dl_dx


def loss_euclidean(y_tilde, y):
  # y = (n,1), y_tilde = (n,1)
  # l = (n,1)
  y_tilde = np.reshape(y_tilde, (-1, 1))
  y = np.reshape(y, (-1, 1))
  dl_dy = y_tilde - y
  l = np.linalg.norm(y - y_tilde) ** 2
  return l, dl_dy


def main_mlp(image_vector_size, learning_rate, iterations, decay_rate, num_samples, x_train):

  num_z = 1000
  num_h = 850
  num_y = image_vector_size

  z = np.random.standard_normal(num_z)
  w1 = np.random.standard_normal((num_h, num_z))
  w2 = np.random.standard_normal((num_y, num_h))
  b1 = np.random.standard_normal((num_h, 1))
  b2 = np.random.standard_normal((num_y, 1))
  Loss = []

  for iter in range(iterations):
    learning_rate = learning_rate * decay_rate
    for i in range(num_samples):
      fc1_out = fc(z, w1, b1)
      h = relu(fc1_out)
      y = fc(h, w2, b2)
      l, dl_dy = loss_euclidean(y, x_train[i])
      Loss.append(l)
      print("Iteration = ", iter, " sample = ", i, " loss = ", l)
      if (iter != 0 and l < 500):
        return z
      dl_dh, dl_dw2, dl_db2 = fc_backward(dl_dy, h, w2, b2, y)
      dl_dfc1_out = relu_backward(dl_dh, fc1_out, h)
      dl_dz, dl_dw1, dl_db1 = fc_backward(dl_dfc1_out, z, w1, b1, fc1_out)
      w1 = w1 - learning_rate * dl_dw1
      b1 = b1 - learning_rate * dl_db1
      w2 = w2 - learning_rate * dl_dw2
      b2 = b2 - learning_rate * dl_db2

      # Regular Autoencoder
      #z = z.reshape(-1, 1) - learning_rate * dl_dz.reshape(-1, 1)

      #Sparse Autoencoder
      l1_param = 1
      abs_z = np.absolute(z)
      sign_z = np.divide(z, abs_z)
      dl_dz = dl_dz.reshape(-1,1) + l1_param * sign_z.reshape(-1,1)
      z = z.reshape(-1, 1) - learning_rate * dl_dz.reshape(-1, 1)
  num_x = num_samples * iterations
  return z, num_x, Loss, w1, b1, w2, b2

def test(z, x_test, w1, b1, w2, b2):
    test_loss = 0
    for i in range(x_test.shape[0]):
      fc1_out = fc(z, w1, b1)
      h = relu(fc1_out)
      y = fc(h, w2, b2)
      l, dl_dy = loss_euclidean(y, x_test[i])
      test_loss += l
    return test_loss

def main():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  image_vector_size = 28
  x_train = shuffle(x_train)
  x_test = shuffle(x_test)
  x_train = x_train.reshape(x_train.shape[0], image_vector_size * image_vector_size)
  x_test = x_test.reshape(x_test.shape[0], image_vector_size * image_vector_size)
  num_samples = x_train.shape[0]

  learning_rate = 0.000001
  decay_rate = 0.1
  iterations = 1

  z, num_x, Loss, w1, b1, w2, b2 = main_mlp(image_vector_size * image_vector_size, learning_rate, iterations, decay_rate, num_samples, x_train)
  print("Encoded feature = ", z)

  x = np.arange(0, num_x)
  Loss = np.asarray(Loss)
  plt.plot(x, Loss)
  plt.xlabel("60k samples, epochs=3")
  plt.ylabel("MSE loss")
  plt.title("MNIST Loss curve - sparse sgd")
  plt.savefig("overcomplete_sparse_sgd_loss_curve_850.jpg")
  plt.show()

  # Training loss (after training)
  train_loss = test(z, x_train, w1, b1, w2, b2)
  print("Training Loss (after training)= ", train_loss)

  # Testing
  test_loss = test(z, x_test, w1, b1, w2, b2)
  print("Testing Loss= ", test_loss)

  # Learned representation after training
  fc1_out = fc(z, w1, b1)
  h = relu(fc1_out)
  y = fc(h, w2, b2)
  representation = y.reshape(image_vector_size, image_vector_size)
  plt.imshow(representation, cmap="gray")
  plt.savefig("overcomplete_sparse_learned_data_representation_850.jpg")
  plt.show()


if __name__ == "__main__":
  main()


