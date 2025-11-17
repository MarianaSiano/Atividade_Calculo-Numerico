import numpy as np
from scipy.linalg import cholesky, solve

# Definição da matriz A e do vetor B
A = np.array([
    [4, 2, -4],
    [2, 10, 4],
    [-4, 4, 9]
])

b = np.array([0, 6, 5])

# a) Verificação das condições para Cholesky
is_symmetric = np.allclose(A, A.T) # Verifica se é simétrica
eigenvalues = np.linalg.eigvals(A) # Obtém autovalores
is_pos_def = np.all(eigenvalues > 0) # Verifica se são positivos

print('A é simétrica? ', is_symmetric)
print('Autovalores de A => ', eigenvalues)
print('A é definida positiva? ', is_pos_def)

# b) Decomposição de Cholesky
G = cholesky(A, lower=True)
print('Matriz G da Decomposição de Cholesky')

# c) Determinante via Cholesky
det_A = np.prod(np.diag(G)) ** 2
print('Determinante de A => ', det_A)

# d) Solução do Sistema Ax = b usando G
# Resolva Gy = b, depois G ^ T x = y
y = solve(G, b, lower=True)
x = solve(G.T, y, lower=False)
print('Solução x => ', x)