import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.ndimage import zoom

class TestUpscale(unittest.TestCase):

    def setUp(self):
        """
        Przygotowuje dane dla testów.
        Ładuje zestaw danych cyfr, dzieli go na dane treningowe i testowe.
        """
        digits = load_digits()
        self.X = digits.data
        self.y = digits.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_add_noise_features(self):
        """
        Testuje, czy funkcja `add_noise_features` poprawnie dodaje kolumny z szumem do danych.
        Sprawdza, czy liczba cech danych wzrasta o oczekiwaną liczbę szumu (100 cech).
        """

        def add_noise_features(X, n_noise_features=100, random_state=42):
            np.random.seed(random_state)
            noise = np.random.normal(0, 1, size=(X.shape[0], n_noise_features))
            return np.hstack((X, noise))

        X_noisy = add_noise_features(self.X_train)
        self.assertEqual(X_noisy.shape[1], self.X_train.shape[1] + 100)

    def test_upscale_images(self):
        """
        Testuje, czy funkcja `upscale_images` poprawnie zwiększa rozdzielczość obrazów
        z oryginalnych wymiarów 8x8 do 16x16. Sprawdza, czy liczba cech po zwiększeniu wynosi 256.
        """

        def upscale_images(X, new_size=(16, 16)):
            return np.array([zoom(img.reshape(8, 8), (new_size[0] / 8, new_size[1] / 8)).flatten() for img in X])

        X_upscaled = upscale_images(self.X_train)
        self.assertEqual(X_upscaled.shape[1], 16 * 16)

    def test_model_with_noise(self):
        """
        Testuje, czy model k-NN może być poprawnie wytrenowany na danych z dodanym szumem.
        Sprawdza, czy model osiąga dokładność powyżej 60% na danych testowych z dodanym szumem.
        """

        def add_noise_features(X, n_noise_features=100, random_state=42):
            np.random.seed(random_state)
            noise = np.random.normal(0, 1, size=(X.shape[0], n_noise_features))
            return np.hstack((X, noise))

        X_noisy = add_noise_features(self.X_train)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_noisy, self.y_train)
        y_pred = knn.predict(add_noise_features(self.X_test))
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.6)

if __name__ == "__main__":
    unittest.main()
