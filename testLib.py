import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

# --- Перевірка NumPy ---
x_np = np.array([[1, 2, 3], [4, 5, 6]])
print("Масив NumPy x:\n{}".format(x_np))

# --- Перевірка SciPy ---
# Створюємо 2D масив NumPy з одиницями по головній діагоналі
eye = np.eye(4)
print("\nМасив NumPy (одинична матриця):\n{}".format(eye))

# --- Перевірка Matplotlib ---
# Генеруємо послідовність чисел від -10 до 10
x_plt = np.linspace(-10, 10, 100)
# Створюємо другий масив за допомогою синуса
y_plt = np.sin(x_plt)
# Функція створює лінійний графік
plt.plot(x_plt, y_plt, marker="x")
plt.title("Тестовий графік Matplotlib")
plt.xlabel("Вісь X")
plt.ylabel("Вісь Y")
plt.show()
