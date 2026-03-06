import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias 

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean((y - self.predict(x)) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - np.sum((y - self.predict(x)) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        weightsGrad = np.ndarray(self.weights.size)
        biasGrad = np.ndarray(self.bias.size)
        weightsGrad = -2 * ((y - self.predict(x)) @ x) / x.shape[0]
        biasGrad = -2 * np.mean(y - self.predict(x))
        return weightsGrad, biasGrad


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(self.weights), np.zeros_like(self.bias)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кузнецов Александр Павлович, ПМ-34"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def fit(model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> None:
        n = y.size
        for i in range(n_iter):
            weightsGrad = np.ndarray(model.weights.size)
            biasGrad = np.ndarray(model.bias.size)
            weightsGrad, biasGrad = model.grad(x, y)
            model.weights -= weightsGrad * lr
            model.bias -= biasGrad * lr