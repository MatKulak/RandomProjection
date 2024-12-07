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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# redukujemy wymiarowość przy użyciu Sparse Random Projection
n_components = 30  # redukujemy do 30 wymiarów
random_proj = SparseRandomProjection(n_components=n_components, random_state=42)
X_train_reduced = random_proj.fit_transform(X_train)
X_test_reduced = random_proj.transform(X_test)

# trenujemy klasyfikator k-NN na zredukowanym zbiorze danych
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_reduced, y_train)

# dokonajumy prognoz i oceniamy model
y_pred = knn.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred)

# raport klasyfikacji i dokładność
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
print(f"Accuracy of k-NN on reduced data: {accuracy:.2f}")

# wyświetlenie oryginalnych obrazów wraz z ich przewidywaniami
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}")
plt.suptitle("Sample Predictions on Test Set", fontsize=16)
plt.show()
