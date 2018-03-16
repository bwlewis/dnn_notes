# From the upcoming book by Taylor Arnold,
# A computational approach to statistical learning.
# Skip to the bottom of this file to see an example.

# Inputs: sizes, a vector giving the size of
#         layer in the neural network, including
#         the input and output layers
# Output: a list containing initialized weights
#         and biases
make_weights = function(sizes) {
  L = length(sizes) - 1
  weights = vector("list", L)
  for (j in seq_len(L)) {
    w = matrix(rnorm(sizes[j] * sizes[j + 1]),
                ncol = sizes[j],
                nrow = sizes[j + 1])
    weights[[j]] = list(w = w,
                         b = rnorm(sizes[j + 1]))
  }
  weights
}

# nonnegtive threshold
# Inputs: v, a vector, matrix or array
# Output: input with negative terms pushed to zero
ReLU = function(v) {
  v[v < 0] = 0
  v
}

# nonnegative threshold gradient
# Inputs: v, a vector, matrix or array
# Output: converts positive terms to one and
#         non-positive terms to zero.
ReLU_p = function(v) {
  p = v * 0
  p[v > 0] = 1
  p
}

# l2 functional gradient
# Inputs: y, response vector; a, predicted
#         response
# Output: gradient of the rmse loss function
rmse_p = function(y, a) {
  (a - y)
}

# forward propagation step
# Inputs: x, one row from the input data; weights,
#         output from make_weights; sigma, the activation
#         function
# Output: list containing the weighted responses (z) and
#         activated responses (a)
f_prop = function(x, weights, sigma) {
  L = length(weights)
  z = vector("list", L)
  a = vector("list", L)
  for (j in seq_len(L)) {
    a_j1 = if(j == 1) x else a[[j - 1L]]
    z[[j]] = weights[[j]]$w %*% a_j1 + weights[[j]]$b
    a[[j]] = if (j != L) sigma(z[[j]]) else z[[j]]
  }
  return(list(z = z, a = a))
}


# backward propagation step
# Inputs: x, one row from the input data; y, one row
#         from the response matrix; weights, output from
#         make_weights; f_obj, output from f_prop;
#         sigma_p, derivative of the activation function;
#         f_p, derivative of the loss function
# Output: gradients
b_prop = function(x, y, weights, f_obj, sigma_p, f_p) {
  z = f_obj$z; a = f_obj$a
  L = length(weights)
  grad_z = vector("list", L)
  grad_w = vector("list", L)
  for (j in rev(seq_len(L))) {
    if (j == L) {
      grad_z[[j]] = f_p(y, a[[j]])
    } else {
      grad_z[[j]] = (t(weights[[j + 1]]$w) %*%
                      grad_z[[j + 1]]) * sigma_p(z[[j]])
    }
    a_j1 = if(j == 1) x else a[[j - 1L]]
    grad_w[[j]] = grad_z[[j]] %*% t(a_j1)
  }

  return(list(grad_z = grad_z, grad_w = grad_w))
}

# Inputs: X, data matrix; y, response; sizes, vector
#         given the number of neurons in each layer of
#         the neural network; epochs, integer number of
#         epochs to complete; rho, learning rate;
#         weights, optional list of starting weights
# Output: the trained weights for the neural network
nn_sgd = function(X, y, sizes, epochs, rho,
                   weights = NULL, progress=TRUE) {

  if (is.null(weights))
    weights = make_weights(sizes)

  for (epoch in seq_len(epochs)) {
    for (i in seq_len(nrow(X))) {
      f_obj = f_prop(X[i,], weights, ReLU)
      b_obj = b_prop(X[i,], y[i,], weights, f_obj,
                      ReLU_p, rmse_p)

      for (j in seq_along(b_obj)) {
        weights[[j]]$b = weights[[j]]$b - rho * b_obj$grad_z[[j]]
        weights[[j]]$w = weights[[j]]$w - rho * b_obj$grad_w[[j]]
      }
    }
    if(progress) {
      yhat = nn_predict(weights, X)
      title = paste("Epoch: ",epoch, "error: ",drop(sqrt(crossprod(yhat - y))))
      plot(X, y, main=title, col="gray")
      i = order(X)
      lines(X[i], yhat[i], col=4, lwd=3)
      Sys.sleep(0.5)
    }
  }

  return(weights)
}

# Inputs: weights, list of neural network weights;
#         X_test, data matrix to predict with
# Output: the predicted values y_hat
nn_predict = function(weights, X_test) {

  p = length(weights[[length(weights)]]$b)
  y_hat = matrix(0, ncol = p, nrow = nrow(X_test))
  for (i in seq_len(nrow(X_test))) {
    a = f_prop(X_test[i,], weights, ReLU)$a
    y_hat[i,] = a[[length(a)]]
  }

  y_hat
}


# Example data and run...
set.seed(1)
X = matrix(runif(1000, min=-1, max=1), ncol=1)
y = X * X + rnorm(1000, sd=0.1)
weights = nn_sgd(X, y, sizes=c(1, 25, 1), epochs=25, rho=0.01)
y_pred = nn_predict(weights, X)

plot(X, y)
i = order(X)
lines(X[i], y_pred[i], col=4, lwd=3)
