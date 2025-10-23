import numpy as np
from sklearn import preprocessing

# Надання позначок вхідних даних
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Створення кодувальника та встановлення відповідності
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Виведення відображення
print("Label mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Перетворення міток за допомогою кодувальника
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Декодування набору чисел за допомогою декодера
encoded_values_to_decode = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values_to_decode)
print("\nEncoded values to decode =", encoded_values_to_decode)
print("Decoded labels =", list(decoded_list))