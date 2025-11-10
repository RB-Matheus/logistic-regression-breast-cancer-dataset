import numpy as np
import matplotlib.pyplot as plt

y_predito = np.linspace(0.01, 0.99, 100)
custo_y_igual_1 = -np.log(y_predito)
custo_y_igual_0 = -np.log(1 - y_predito)

plt.figure(figsize=(10, 6))
plt.plot(y_predito, custo_y_igual_1, label='Erro para y = 1', color='blue')
plt.plot(y_predito, custo_y_igual_0, label='Erro para y = 0', color='red', linestyle='--')
plt.title('Cálculo de custo do erro para y = 1 e y = 0')
plt.xlabel(r'$\hat{y} = \sigma(z)$')
plt.ylabel('Custo BCE')
plt.ylim(0, 5)
plt.grid(True)
plt.legend()
plt.show()