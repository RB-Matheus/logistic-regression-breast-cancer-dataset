import numpy as np
import matplotlib.pyplot as plt

x0 = np.linspace(-10, 10, 400)

# Saída da Softmax para y0 dado n entradas (as outras foram fixadas em 0)
def softmax_y0(x0, n):
    return np.exp(x0) / (np.exp(x0) + (n - 1))

# Cálculo das curvas quando o número de entradas valee 2, 3 e 10, respectivamente
y0_2 = softmax_y0(x0, 2)
y0_3 = softmax_y0(x0, 3)
y0_10 = softmax_y0(x0, 10)

# Plotagem
plt.figure(figsize=(8, 5))
plt.plot(x0, y0_2, 'g', label='$y_0$ da Softmax de 2 entradas')
plt.plot(x0, y0_3, 'b', label='$y_0$ da Softmax de 3 entradas')
plt.plot(x0, y0_10, 'r', label='$y_0$ da Softmax de 10 entradas')

# Estabelecimento de uma linha de referência onde x0 = 0
plt.axvline(0, color='k', linestyle='--', label='$x_0 = x_1 = ... = 0$')

# Colocando os rótulos dos eixos e as legendas
plt.title('Saída da Softmax para $y_0$ com demais entradas fixas em 0', fontsize=14)
plt.xlabel('$x_0$ (variando a entrada)', fontsize=12)
plt.ylabel('$y_0$ (saída da Softmax)', fontsize=12)
plt.legend()
plt.show()