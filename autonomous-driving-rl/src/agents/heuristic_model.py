"""Small supervised MLP for local cost-to-go residuals, using repo-installed NumPy.

Weights are genuinely optimized with backpropagation and Adam. The prediction is
not certified admissible; graph validation reports overestimation and path costs.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from src.agents.planning_graph import FEATURE_NAMES


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HeuristicMLP:
    def __init__(self, hidden_units=32, seed=101):
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, np.sqrt(1 / len(FEATURE_NAMES)), (len(FEATURE_NAMES), hidden_units)),
            "b1": np.zeros(hidden_units),
            "w2": rng.normal(0, np.sqrt(1 / hidden_units), (hidden_units, 1)),
            "b2": np.zeros(1),
        }
        self.mean = np.zeros(len(FEATURE_NAMES))
        self.std = np.ones(len(FEATURE_NAMES))
        self.metadata = {}

    def predict_residual(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid heuristic feature matrix")
        normalized = np.clip((x - self.mean) / self.std, -10, 10)
        hidden = np.tanh(normalized @ self.params["w1"] + self.params["b1"])
        return (hidden @ self.params["w2"] + self.params["b2"])[:, 0]

    def heuristic(self, graph):
        residual = np.maximum(0.0, self.predict_residual(graph.features)) * graph.scale_m
        h = graph.euclidean_heuristic() + residual
        h[graph.goal] = 0.0
        return h

    def loss_and_grad(self, features, targets):
        x = np.clip((features - self.mean) / self.std, -10, 10)
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        prediction = (hidden @ self.params["w2"] + self.params["b2"])[:, 0]
        error = prediction - targets
        output_grad = (2 * error / len(error))[:, None]
        hidden_grad = (output_grad @ self.params["w2"].T) * (1 - hidden * hidden)
        gradients = dict(w1=x.T @ hidden_grad, b1=hidden_grad.sum(axis=0),
                         w2=hidden.T @ output_grad, b2=output_grad.sum(axis=0))
        return float(np.mean(error ** 2)), gradients

    def fit(self, train_x, train_y, holdout_x, holdout_y, *, epochs=40, batch_size=128,
            learning_rate=.003, seed=101, on_epoch=None):
        for x, y in ((train_x, train_y), (holdout_x, holdout_y)):
            if len(x) == 0 or x.shape != (len(y), len(FEATURE_NAMES)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("Training and grouped holdout data must be finite and nonempty")
        self.mean = train_x.mean(axis=0)
        self.std = np.maximum(train_x.std(axis=0), 1e-3)
        rng = np.random.default_rng(seed)
        first = {key: np.zeros_like(value) for key, value in self.params.items()}
        second = {key: np.zeros_like(value) for key, value in self.params.items()}
        best_loss = float("inf")
        best_params = None
        logs = []
        step = 0
        for epoch in range(epochs + 1):
            if epoch:
                order = rng.permutation(len(train_x))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    _, gradients = self.loss_and_grad(train_x[indices], train_y[indices])
                    norm = np.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                    step += 1
                    for key, gradient in gradients.items():
                        gradient = gradient * min(1.0, 5.0 / max(norm, 1e-12))
                        first[key] = .9 * first[key] + .1 * gradient
                        second[key] = .999 * second[key] + .001 * gradient * gradient
                        self.params[key] -= learning_rate * (first[key] / (1 - .9 ** step)) / (
                            np.sqrt(second[key] / (1 - .999 ** step)) + 1e-8)
            train_loss = float(np.mean((self.predict_residual(train_x) - train_y) ** 2))
            holdout_loss = float(np.mean((self.predict_residual(holdout_x) - holdout_y) ** 2))
            if not np.isfinite(train_loss + holdout_loss):
                raise ValueError("Non-finite training loss")
            record = dict(epoch=epoch, train_mse=train_loss, internal_holdout_mse=holdout_loss)
            logs.append(record)
            if holdout_loss < best_loss:
                best_loss = holdout_loss
                best_params = {key: value.copy() for key, value in self.params.items()}
                best_epoch = epoch
            if on_epoch:
                on_epoch(record)
        self.params = best_params
        self.metadata.update(best_epoch=best_epoch, initial_train_mse=logs[0]["train_mse"],
                             initial_holdout_mse=logs[0]["internal_holdout_mse"], best_holdout_mse=best_loss,
                             final_train_mse=float(np.mean((self.predict_residual(train_x) - train_y) ** 2)),
                             optimizer_steps=step)
        return logs

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.update(metadata)
        self.metadata.update(format_version="heuristic_mlp_v1", feature_names=list(FEATURE_NAMES),
                             admissibility_certified=False)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, **self.params, mean=self.mean, std=self.std,
                                metadata=np.array(json.dumps(self.metadata, sort_keys=True)))
        temporary.replace(path)
        return file_sha256(path)

    @classmethod
    def load(cls, path, expected_sha256, expected_planner_fingerprint):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Checkpoint SHA-256 mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("format_version") != "heuristic_mlp_v1" or metadata.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("Unsupported checkpoint feature schema")
            if metadata.get("planner_fingerprint") != expected_planner_fingerprint:
                raise ValueError("Checkpoint was trained for a different planner configuration")
            hidden_units = int(data["b1"].shape[0])
            if not 1 <= hidden_units <= 256:
                raise ValueError("Invalid model size")
            model = cls(hidden_units=hidden_units)
            for key in model.params:
                array = np.asarray(data[key], dtype=np.float64)
                if array.shape != model.params[key].shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"Invalid checkpoint tensor: {key}")
                model.params[key] = array.copy()
            model.mean = np.asarray(data["mean"], dtype=np.float64).copy()
            model.std = np.asarray(data["std"], dtype=np.float64).copy()
            if (model.mean.shape != (len(FEATURE_NAMES),) or model.std.shape != model.mean.shape
                    or not np.all(np.isfinite(model.mean)) or not np.all(np.isfinite(model.std)) or np.any(model.std <= 0)):
                raise ValueError("Invalid checkpoint normalization")
            model.metadata = metadata
        return model
