import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

# Завантаження даних
# Переконайтесь, що файл 'data_metrics.csv' знаходиться в тій самій папці
df = pd.read_csv('data_metrics.csv')

# Додавання прогнозованих міток з порогом 0.5
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')

# --- Ваші власні функції для метрик з прізвищем "azarov" ---

def find_TP(y_true, y_pred):
    # Розрахунок True Positives
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    # Розрахунок False Negatives
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    # Розрахунок False Positives
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    # Розрахунок True Negatives
    return sum((y_true == 0) & (y_pred == 0))

def azarov_confusion_matrix(y_true, y_pred):
    """Обчислює матрицю помилок."""
    TN = find_TN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    TP = find_TP(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def azarov_accuracy_score(y_true, y_pred):
    """Обчислює точність (Accuracy)."""
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def azarov_recall_score(y_true, y_pred):
    """Обчислює повноту (Recall / Sensitivity)."""
    TP, FN = find_TP(y_true, y_pred), find_FN(y_true, y_pred)
    if (TP + FN) == 0:
        return 0
    return TP / (TP + FN)

def azarov_precision_score(y_true, y_pred):
    """Обчислює точність (Precision)."""
    TP, FP = find_TP(y_true, y_pred), find_FP(y_true, y_pred)
    if (TP + FP) == 0:
        return 0
    return TP / (TP + FP)

def azarov_f1_score(y_true, y_pred):
    """Обчислює F1-міру."""
    recall = azarov_recall_score(y_true, y_pred)
    precision = azarov_precision_score(y_true, y_pred)
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


# --- Перевірка відповідності результатів ---
print("--- Перевірка confusion_matrix ---")
print("azarov_confusion_matrix (RF):\n", azarov_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print("Sklearn confusion_matrix (RF):\n", confusion_matrix(df.actual_label.values, df.predicted_RF.values))
# Перевірка з assert
assert np.array_equal(azarov_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print("Перевірка пройдена успішно!")
print("\n" + "="*40 + "\n")


# --- Порівняння результатів для порогу 0.5 ---
print('--- Scores with threshold = 0.5 ---')
print('Accuracy RF: %.3f' % azarov_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Recall RF: %.3f' % azarov_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Precision RF: %.3f' % azarov_precision_score(df.actual_label.values, df.predicted_RF.values))
print('F1 RF: %.3f' % azarov_f1_score(df.actual_label.values, df.predicted_RF.values))
print("\n" + "="*40 + "\n")

# --- Порівняння результатів для порогу 0.25 ---
print('--- Scores with threshold = 0.25 ---')
predicted_RF_025 = (df.model_RF >= 0.25).astype('int')
print('Accuracy RF: %.3f' % azarov_accuracy_score(df.actual_label.values, predicted_RF_025.values))
print('Recall RF: %.3f' % azarov_recall_score(df.actual_label.values, predicted_RF_025.values))
print('Precision RF: %.3f' % azarov_precision_score(df.actual_label.values, predicted_RF_025.values))
print('F1 RF: %.3f' % azarov_f1_score(df.actual_label.values, predicted_RF_025.values))
print("\n" + "="*40 + "\n")

# Висновок: Зниження порогу збільшило повноту (Recall), але зменшило точність (Precision).


# --- ROC Curve та AUC ---
print('--- Побудова ROC-кривої ---')
fpr_RF, tpr_RF, _ = roc_curve(df.actual_label, df.model_RF)
fpr_LR, tpr_LR, _ = roc_curve(df.actual_label, df.model_LR)
auc_RF = roc_auc_score(df.actual_label, df.model_RF)
auc_LR = roc_auc_score(df.actual_label, df.model_LR)

plt.figure(figsize=(10, 7))
plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-.', label='Perfect model')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()

# Висновок для звіту:
# Модель RF є кращою, оскільки її крива знаходиться вище і лівіше,
# а площа під кривою (AUC) значно більша, ніж у моделі LR.
# Це означає, що RF краще розрізняє класи при будь-якому порозі.