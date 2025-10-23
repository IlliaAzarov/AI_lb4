import numpy as np
from sklearn import preprocessing

# --- Дані для Варіанту №1 ---
# Вхідні дані з Таблиці 1
input_data = np.array([
    [4.3, -9.9, -3.5],
    [-2.9, 4.1, 3.3],
    [-2.2, 8.8, -6.1],
    [3.9, 1.4, 2.2]
])

# Поріг бінаризації з Таблиці 1
binarization_threshold = 2.2

print("--- Вхідні дані (Варіант 1) ---")
print(input_data)
print("\n" + "="*40 + "\n")

# 1. Бінарізація даних
# Використовуємо поріг 2.2 для вашого варіанту
data_binarized = preprocessing.Binarizer(threshold=binarization_threshold).transform(input_data)
print("Binarized data (Поріг = {}):\n".format(binarization_threshold), data_binarized)
print("\n" + "="*40 + "\n")


# 2. Виключення середнього
print("--- Виключення середнього ---")
print("BEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))
print("\n" + "="*40 + "\n")


# 3. Масштабування MinMax
print("--- Масштабування MinMax (діапазон 0-1) ---")
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("Min max scaled data:\n", data_scaled_minmax)
print("\n" + "="*40 + "\n")


# 4. Нормалізація даних
print("--- Нормалізація ---")
# L1-нормалізація
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
print("L1 normalized data:\n", data_normalized_l1)

# L2-нормалізація
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL2 normalized data:\n", data_normalized_l2)