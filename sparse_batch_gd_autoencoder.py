import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist


def fc(x, w, b):
  # x = (m,1), label = (n,1), w = (n,m), b = (n,1)
  # y = (n,1)
  z = np.reshape(x, (-1,1))
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
  # dl_dw = np.matmul(dl_dy, np.reshape(x, (1, -1)))
  #dl_db = dl_dy

  # Regularized
  l2_param = 1
  dl_dw = np.matmul(dl_dy, np.reshape(x, (1, -1))) + (l2_param * w)
  # dl_db = dl_dy * dyj_dbj --> dyj_dbj = 1
  dl_db = dl_dy.reshape(-1, 1) + (l2_param * b.reshape(-1, 1))

  return dl_dx, dl_dw, dl_db

def relu(x):
  y = np.copy(x)
  # y[y<0] *= 0.005
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
  y_tilde = np.reshape(y_tilde, (-1,1))
  y = np.reshape(y, (-1, 1))
  dl_dy = y_tilde - y
  l = np.linalg.norm(y - y_tilde) ** 2
  return l, dl_dy

def main_mlp():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  image_vector_size = 28 * 28
  x_train = x_train.reshape(x_train.shape[0], image_vector_size)
  x_test = x_test.reshape(x_test.shape[0], image_vector_size)
  num_samples = x_train.shape[0]

  # Hyperparameters
  learning_rate = 0.000001
  decay_rate = 0.1
  iterations = 3
  batch_size = 30
  num_z = 10
  num_h = 300
  num_y = image_vector_size

  z = np.random.standard_normal(num_z)
  w1 = np.random.standard_normal((num_h, num_z))
  w2 = np.random.standard_normal((num_y, num_h))
  b1 = np.random.standard_normal((num_h, 1))
  b2 = np.random.standard_normal((num_y, 1))

  num_batches = int(num_samples / batch_size)
  print("Total batches = ", num_batches)

  for iter in range(iterations):
    learning_rate = learning_rate * decay_rate
    for current_batch in range(num_batches):
      dL_dz = np.zeros((num_z,1))
      dL_dw1 = np.zeros((num_h, num_z))
      dL_dw2 = np.zeros((num_y, num_h))
      dL_db1 = np.zeros((num_h, 1))
      dL_db2 = np.zeros((num_y, 1))
      loss = 0
      for i in range(batch_size):
        sample_number = current_batch * batch_size + i
        fc1_out = fc(z, w1, b1)
        h = relu(fc1_out)
        y = fc(h, w2, b2)
        l, dl_dy = loss_euclidean(y, x_train[sample_number])
        if(iter != 0 and l < 500):
          return z
        dl_dh, dl_dw2, dl_db2 = fc_backward(dl_dy, h, w2, b2, y)
        dl_dfc1_out = relu_backward(dl_dh, fc1_out, h)
        dl_dz, dl_dw1, dl_db1 = fc_backward(dl_dfc1_out, z, w1, b1, fc1_out)
        dL_dw1 += dl_dw1
        dL_dw2 += dl_dw2
        dL_db1 += dl_db1
        dL_db2 += dl_db2
        dL_dz += dl_dz
        loss += l
      w1 = w1 - learning_rate * dL_dw1
      b1 = b1 - learning_rate * dL_db1
      w2 = w2 - learning_rate * dL_dw2
      b2 = b2 - learning_rate * dL_db2

      # Regular Autoencoder
      # z = z.reshape(-1, 1) - learning_rate * dL_dz.reshape(-1, 1)

      # Sparse Autoencoder
      l1_param = 100
      abs_z = np.absolute(z)
      sign_z = np.divide(z, abs_z)
      dL_dz = dL_dz.reshape(-1, 1) + batch_size * l1_param * sign_z.reshape(-1, 1)
      z = z.reshape(-1, 1) - learning_rate * dL_dz.reshape(-1, 1)

      print("Iteration = ", iter, " batch = ", current_batch, " loss = ", loss)
  return z

z = main_mlp()
print("Encoded feature = ", z)

