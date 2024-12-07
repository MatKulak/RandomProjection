import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# tu wczytujemy dane
digits = datasets.load_digits()
X, y = digits.data, digits.target

# dzielimy na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# trenujemy klasyfikator k-NN na nie zredukowanym zbiorze danych
knn_original = KNeighborsClassifier(n_neighbors=5)
knn_original.fit(X_train, y_train)

# dokonajumy prognoz i oceniamy model
y_pred_original = knn_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

# raport klasyfikacji i dokładność na nie zredukowanych danych
report_original = classification_report(y_test, y_pred_original)
print("Classification Report (Original Data):\n", report_original)
print(f"Accuracy of k-NN on original data: {accuracy_original:.2f}")

fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred_original[i]}")
plt.suptitle("Sample Predictions on Test Set (Original Data)", fontsize=16)
plt.show()
