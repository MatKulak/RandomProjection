import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# dodajemy szum
def add_noise_features(X, n_noise_features=100, random_state=42):
    np.random.seed(random_state)
    noise = np.random.normal(0, 1, size=(X.shape[0], n_noise_features))
    X_extended = np.hstack((X, noise))
    return X_extended

# zwiekszamy rozdzielczość obrazu
def upscale_images(X, new_size=(16, 16)):
    upscaled_images = np.array([zoom(img.reshape(8, 8), (new_size[0]/8, new_size[1]/8)).flatten() for img in X])
    return upscaled_images

# zwiekszamy cechy
X_train_noisy = add_noise_features(X_train, n_noise_features=100)
X_test_noisy = add_noise_features(X_test, n_noise_features=100)
X_train_upscaled = upscale_images(X_train)
X_test_upscaled = upscale_images(X_test)

# klasyfikacje danych z szumem
print("\n=== Klasyfikacja na danych z szumem ===")
knn_noisy = KNeighborsClassifier(n_neighbors=5)
knn_noisy.fit(X_train_noisy, y_train)
y_pred_noisy = knn_noisy.predict(X_test_noisy)
accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
print(f"Dokładność: {accuracy_noisy:.2f}")
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_noisy))

# klasyfikacje danych z interpolacją
print("\n=== Klasyfikacja na danych z interpolacją (16x16 pikseli) ===")
knn_upscaled = KNeighborsClassifier(n_neighbors=5)
knn_upscaled.fit(X_train_upscaled, y_train)
y_pred_upscaled = knn_upscaled.predict(X_test_upscaled)
accuracy_upscaled = accuracy_score(y_test, y_pred_upscaled)
print(f"Dokładność: {accuracy_upscaled:.2f}")
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_upscaled))

# wizualizacja dla danych z interpolacją
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_test_upscaled[i].reshape(16, 16), cmap='gray')
    ax.set_title(f"Pred: {y_pred_upscaled[i]}")
plt.suptitle("Predykcje na danych z interpolacją (16x16)", fontsize=16)
plt.show()

# wizualizacja dla danych z szumem
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred_noisy[i]}")
plt.suptitle("Predykcje na danych z dodanym szumem (8x8 + szum)", fontsize=16)
plt.show()
