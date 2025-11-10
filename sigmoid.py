import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoide(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\frac{1}{1 + e^{-z}}$')
plt.xlabel('z')
plt.ylabel(r'$\hat{y} = \sigma(z)$')
plt.title('Sigmoide')
plt.grid(True)
plt.legend(fontsize=16)
plt.show()