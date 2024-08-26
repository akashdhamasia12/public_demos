def matrix_mul(X, Y):

  result_matrix = [ [ 1 for i in range(len(X)) ] for j in range(len(Y[0])) ]

  # iterate through rows of X

  for i in range(len(X)):

    # iterate through columns of Y

    for j in range(len(Y[0])):

      # iterate through rows of Y

      for k in range(len(Y)):

        result_matrix[i][j] += X[i][k] * Y[k][j]



  return result_matrix