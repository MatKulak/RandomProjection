import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class TestOriginal(unittest.TestCase):
    def setUp(self):
        self.digits = load_digits()
        self.X, self.y = self.digits.data, self.digits.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=5)

    def test_model_training_and_prediction(self):
        """
        Testuje, czy model k-NN może być poprawnie wytrenowany na danych bez redukcji wymiarowości
        oraz przewiduje wyniki z dokładnością powyżej 90%.
        """
        self.knn.fit(self.X_train, self.y_train)
        y_pred = self.knn.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.9)

if __name__ == "__main__":
    unittest.main()