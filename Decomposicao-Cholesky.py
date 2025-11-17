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
is_symmetric = np.allclose(A.T) # Verifica se é simétrica
eigenvalues = np.linalg.eigvals(A) # Obtém autovalores
is_pos_def = np.all(eigenvalues > 0) # Verifica se são positivos

print('A é simétrica? ', is_symmetric)
print('Autovalores de A => ', eigenvalues)
print('A é definida positiva? ', is_pos_def)