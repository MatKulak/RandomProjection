import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def add_noise_features(X, n_noise_features=100, random_state=42):
    np.random.seed(random_state)
    noise = np.random.normal(0, 1, size=(X.shape[0], n_noise_features))  # Generowanie szumu
    X_extended = np.hstack((X, noise))  # Dodanie szumu jako nowych kolumn
    return X_extended

X_train_noisy = add_noise_features(X_train, n_noise_features=100)
X_test_noisy = add_noise_features(X_test, n_noise_features=100)

def upscale_images(X, new_size=(16, 16)):
    upscaled_images = np.array([zoom(img.reshape(8, 8), (new_size[0]/8, new_size[1]/8)).flatten() for img in X])
    return upscaled_images

X_train_upscaled = upscale_images(X_train)
X_test_upscaled = upscale_images(X_test)

def apply_random_projection(X_train, X_test, n_components, projection_type="gaussian"):
    if projection_type == "gaussian":
        projector = GaussianRandomProjection(n_components=n_components, random_state=42)
    elif projection_type == "sparse":
        projector = SparseRandomProjection(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unsupported projection_type. Use 'gaussian' or 'sparse'.")
    X_train_projected = projector.fit_transform(X_train)
    X_test_projected = projector.transform(X_test)
    return X_train_projected, X_test_projected

X_train_noisy_proj, X_test_noisy_proj = apply_random_projection(X_train_noisy, X_test_noisy, n_components=50, projection_type="gaussian")

X_train_upscaled_proj, X_test_upscaled_proj = apply_random_projection(X_train_upscaled, X_test_upscaled, n_components=50, projection_type="gaussian")

print("\n=== Klasyfikacja na danych z szumem po Random Projection ===")
knn_noisy_proj = KNeighborsClassifier(n_neighbors=5)
knn_noisy_proj.fit(X_train_noisy_proj, y_train)
y_pred_noisy_proj = knn_noisy_proj.predict(X_test_noisy_proj)
accuracy_noisy_proj = accuracy_score(y_test, y_pred_noisy_proj)
print(f"Dokładność: {accuracy_noisy_proj:.2f}")
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_noisy_proj))

print("\n=== Klasyfikacja na danych z interpolacją po Random Projection ===")
knn_upscaled_proj = KNeighborsClassifier(n_neighbors=5)
knn_upscaled_proj.fit(X_train_upscaled_proj, y_train)
y_pred_upscaled_proj = knn_upscaled_proj.predict(X_test_upscaled_proj)
accuracy_upscaled_proj = accuracy_score(y_test, y_pred_upscaled_proj)
print(f"Dokładność: {accuracy_upscaled_proj:.2f}")
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_upscaled_proj))


fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred_noisy_proj[i]}")
plt.suptitle("Predykcje na danych z szumem po Random Projection", fontsize=16)
plt.show()

fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred_upscaled_proj[i]}")
plt.suptitle("Predykcje na danych z interpolacją po Random Projection", fontsize=16)
plt.show()
