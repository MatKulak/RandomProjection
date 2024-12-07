import unittest
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class TestFile1(unittest.TestCase):
    def setUp(self):
        self.digits = load_digits()
        self.X, self.y = self.digits.data, self.digits.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)
        self.n_components = 30
        self.random_proj = SparseRandomProjection(n_components=self.n_components, random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=5)

    def test_dimension_reduction(self):
        """
        Testuje, czy po redukcji wymiarowości liczba cech danych wynosi dokładnie `n_components`.
        Sprawdza, czy redukcja wymiarowości za pomocą SparseRandomProjection działa poprawnie.
        """
        X_train_reduced = self.random_proj.fit_transform(self.X_train)
        self.assertEqual(X_train_reduced.shape[1], self.n_components)

    def test_model_training_and_prediction(self):
        """
        Testuje, czy model k-NN może być poprawnie wytrenowany i przewiduje wyniki
        z dokładnością powyżej 80% na danych zredukowanych przy użyciu SparseRandomProjection.
        """
        X_train_reduced = self.random_proj.fit_transform(self.X_train)
        X_test_reduced = self.random_proj.transform(self.X_test)
        self.knn.fit(X_train_reduced, self.y_train)
        y_pred = self.knn.predict(X_test_reduced)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()

