import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# --- Оцінка SVM класифікатора ---
# Створення SVM класифікатора
classifier_svm = SVC(kernel='linear', C=1.0) # Можна спробувати 'rbf', 'poly'

print("--- Support Vector Machine (SVM) ---")
num_folds = 3
# Accuracy
accuracy_values_svm = cross_val_score(classifier_svm, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values_svm.mean(), 2)) + "%")
# Precision
precision_values_svm = cross_val_score(classifier_svm, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values_svm.mean(), 2)) + "%")
# Recall
recall_values_svm = cross_val_score(classifier_svm, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values_svm.mean(), 2)) + "%")
# F1
f1_values_svm = cross_val_score(classifier_svm, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values_svm.mean(), 2)) + "%")


# --- Оцінка Naive Bayes класифікатора для порівняння ---
classifier_nb = GaussianNB()

print("\n--- Naive Bayes (for comparison) ---")
# Accuracy
accuracy_values_nb = cross_val_score(classifier_nb, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values_nb.mean(), 2)) + "%")
# Precision
precision_values_nb = cross_val_score(classifier_nb, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values_nb.mean(), 2)) + "%")
# Recall
recall_values_nb = cross_val_score(classifier_nb, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values_nb.mean(), 2)) + "%")
# F1
f1_values_nb = cross_val_score(classifier_nb, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values_nb.mean(), 2)) + "%")

# Висновок для звіту: Порівняйте отримані показники. Якщо показники SVM вищі, то він є кращою моделлю для цих даних,
# оскільки він ефективніше будує розділову гіперплощину між класами.