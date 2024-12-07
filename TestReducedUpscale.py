import unittest
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics import accuracy_score


class TestReducedUpscale(unittest.TestCase):
    def setUp(self):
        self.digits = load_digits()
        self.X, self.y = self.digits.data, self.digits.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def test_add_noise(self):
        """
        Testuje, czy funkcja `add_noise_features` dodaje poprawną liczbę cech (szumu)
        do danych. Sprawdza, czy wymiar danych zwiększa się o oczekiwaną liczbę cech.
        """

        def add_noise_features(X, n_noise_features=100, random_state=42):
            np.random.seed(random_state)
            noise = np.random.normal(0, 1, size=(X.shape[0], n_noise_features))
            return np.hstack((X, noise))

        X_noisy = add_noise_features(self.X_train)
        self.assertEqual(X_noisy.shape[1], self.X_train.shape[1] + 100)

    def test_random_projection(self):
        """
        Testuje, czy funkcja `apply_random_projection` poprawnie redukuje wymiar danych
        do określonej liczby komponentów. Sprawdza, czy liczba cech po projekcji wynosi `n_components`.
        """

        def apply_random_projection(X_train, n_components=50, projection_type="gaussian"):
            if projection_type == "gaussian":
                projector = GaussianRandomProjection(n_components=n_components, random_state=42)
            elif projection_type == "sparse":
                projector = SparseRandomProjection(n_components=n_components, random_state=42)
            else:
                raise ValueError("Invalid projection type.")
            return projector.fit_transform(X_train)

        X_projected = apply_random_projection(self.X_train, n_components=50)
        self.assertEqual(X_projected.shape[1], 50)

    def test_model_training_with_noise(self):
        """
        Testuje, czy model k-NN może być poprawnie wytrenowany na danych z dodanym szumem.
        Sprawdza, czy model osiąga dokładność powyżej 70% na danych testowych z dodanym szumem.
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
        self.assertGreater(accuracy, 0.7)

if __name__ == "__main__":
    unittest.main()
